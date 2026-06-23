//! Background job processing system for offloading asynchronous, non-blocking tasks.
//!
//! Provides a thread-safe, priority-aware background job pool with task priorities,
//! exponential backoff retries, execution conditions (delayed execution), graceful
//! shutdown, and disk-based state persistence.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{error, info};
use uuid::Uuid;

use crate::runtime::store::RunEventRecord;
use crate::runtime::{AgentEvent, RunId, RuntimeStore};

#[cfg(feature = "queue")]
use crate::queue::EventPublisher;

/// Errors produced during background job execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JobError {
    /// Failed to append event or interact with the storage.
    #[error("storage error: {0}")]
    Storage(#[from] crate::error::StorageError),

    /// Failed due to a runtime error.
    #[error("runtime error: {0}")]
    Runtime(#[from] crate::runtime::RuntimeError),

    /// Failed to publish event to the external queue.
    #[cfg(feature = "queue")]
    #[error("queue error: {0}")]
    Queue(#[from] crate::queue::QueueError),

    /// Custom job execution failure.
    #[error("job execution failed: {0}")]
    Execution(String),
}

/// Priority levels for background jobs.
///
/// Jobs with higher priority levels are executed first. For jobs with
/// identical priority levels, earlier created jobs are executed first (FIFO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JobPriority {
    /// Low priority (e.g. routine cleanup, logs).
    Low = 0,
    /// Normal priority (e.g. routine event persistence).
    Normal = 1,
    /// High priority (e.g. external queue notification).
    High = 2,
    /// Critical priority (e.g. system state synchronization).
    Critical = 3,
}

/// The specific task logic represented by a background job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
#[non_exhaustive]
pub enum JobType {
    /// Persists an event to the run store.
    PersistEvent {
        /// The associated run ID.
        run_id: RunId,
        /// The event to persist.
        event: AgentEvent,
    },

    /// Publishes an event to NATS or Redis Streams.
    #[cfg(feature = "queue")]
    PublishExternalEvent {
        /// The event to publish.
        event: AgentEvent,
    },

    /// Dummy task for testing.
    TestJob {
        /// Test payload message.
        message: String,
    },
}

/// Execution conditions governing when a job is allowed to run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobConditions {
    /// Earliest time at which the job can be executed.
    pub run_after: Option<DateTime<Utc>>,
}

/// A background job unit of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundJob {
    /// Unique identifier for this job.
    pub id: Uuid,
    /// The priority of the job.
    pub priority: JobPriority,
    /// Number of times the job has been retried.
    pub retries: usize,
    /// Maximum number of retries before discarding the job.
    pub max_retries: usize,
    /// The specific task to execute.
    pub job_type: JobType,
    /// Execution constraints.
    pub conditions: JobConditions,
    /// Timestamp when this job was first created.
    pub created_at: DateTime<Utc>,
}

impl Ord for BackgroundJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => {
                // Invert created_at so earlier timestamp is greater (popped first in BinaryHeap)
                other.created_at.cmp(&self.created_at)
            }
            ord => ord,
        }
    }
}

impl PartialOrd for BackgroundJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BackgroundJob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for BackgroundJob {}

/// A thread-safe background job pool that schedules and executes jobs based on priority and conditions.
pub struct BackgroundJobPool {
    store: Arc<RuntimeStore>,
    #[cfg(feature = "queue")]
    event_publisher: std::sync::RwLock<Option<Arc<dyn EventPublisher>>>,
    jobs: Mutex<BinaryHeap<BackgroundJob>>,
    notify: Notify,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
    worker_handle: Mutex<Option<JoinHandle<()>>>,
    persistence_path: Option<String>,
    completed_tx: tokio::sync::broadcast::Sender<(Uuid, Result<JobType, String>)>,
}

impl BackgroundJobPool {
    /// Creates a new background job pool.
    #[must_use]
    pub fn new(
        store: Arc<RuntimeStore>,
        #[cfg(feature = "queue")] event_publisher: Option<Arc<dyn EventPublisher>>,
        persistence_path: Option<String>,
    ) -> Arc<Self> {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (completed_tx, _) = tokio::sync::broadcast::channel(100);
        Arc::new(Self {
            store,
            #[cfg(feature = "queue")]
            event_publisher: std::sync::RwLock::new(event_publisher),
            jobs: Mutex::new(BinaryHeap::new()),
            notify: Notify::new(),
            shutdown_tx,
            shutdown_rx,
            worker_handle: Mutex::new(None),
            persistence_path,
            completed_tx,
        })
    }

    /// Subscribes to the broadcast stream of executed job results.
    #[must_use]
    pub fn subscribe_completed(
        &self,
    ) -> tokio::sync::broadcast::Receiver<(Uuid, Result<JobType, String>)> {
        self.completed_tx.subscribe()
    }

    /// Sets/updates the external event publisher.
    #[cfg(feature = "queue")]
    pub fn set_event_publisher(&self, publisher: Arc<dyn EventPublisher>) {
        if let Ok(mut lock) = self.event_publisher.write() {
            *lock = Some(publisher);
        }
    }

    /// Helper to schedule a job.
    pub async fn schedule(
        &self,
        priority: JobPriority,
        job_type: JobType,
        conditions: JobConditions,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let job = BackgroundJob {
            id,
            priority,
            retries: 0,
            max_retries: 3,
            job_type,
            conditions,
            created_at: Utc::now(),
        };
        self.enqueue(job).await;
        id
    }

    /// Starts the background worker thread/task.
    ///
    /// # Panics
    ///
    /// Panics if the worker is already started.
    pub fn start(self: &Arc<Self>) {
        let Ok(mut handle_lock) = self.worker_handle.try_lock() else {
            panic!("worker handle lock poisoned");
        };
        assert!(
            handle_lock.is_none(),
            "background job pool worker already started"
        );

        let pool_clone = Arc::clone(self);
        let handle = tokio::spawn(async move {
            run_worker_loop(pool_clone).await;
        });
        *handle_lock = Some(handle);
    }

    /// Enqueues a job into the pool.
    pub async fn enqueue(&self, job: BackgroundJob) {
        {
            let mut heap = self.jobs.lock().await;
            heap.push(job);
        }
        self.notify.notify_one();
    }

    /// Shuts down the background job pool gracefully, joining the worker and persisting pending jobs.
    pub async fn shutdown(&self) {
        info!("shutting down background job pool...");
        let _ = self.shutdown_tx.send(true);
        self.notify.notify_one();

        let mut handle_lock = self.worker_handle.lock().await;
        if let Some(handle) = handle_lock.take() {
            if let Err(e) = handle.await {
                error!(error = ?e, "background job pool worker joined with error");
            }
        }

        // Persist any unexecuted/pending jobs to disk
        if let Some(path) = &self.persistence_path {
            let pending_jobs: Vec<BackgroundJob>;
            {
                let heap = self.jobs.lock().await;
                pending_jobs = heap.clone().into_sorted_vec();
            }

            if pending_jobs.is_empty() {
                // If there are no pending jobs, clean up the persistence file if it exists
                if std::path::Path::new(path).exists() {
                    let _ = fs::remove_file(path);
                }
            } else {
                info!(count = pending_jobs.len(), path = %path, "persisting pending background jobs...");
                if let Some(parent) = std::path::Path::new(path).parent() {
                    let _ = fs::create_dir_all(parent);
                }
                match serde_json::to_string_pretty(&pending_jobs) {
                    Ok(json) => {
                        if let Err(e) = fs::write(path, json) {
                            error!(error = %e, "failed to write background jobs to disk");
                        } else {
                            info!("persisted pending background jobs successfully");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "failed to serialize background jobs");
                    }
                }
            }
        }
        info!("background job pool shutdown complete");
    }

    /// Loads persisted jobs from the configured persistence file and enqueues them.
    pub async fn load_persisted_jobs(&self) {
        let Some(path) = &self.persistence_path else {
            return;
        };

        if !std::path::Path::new(path).exists() {
            return;
        }

        info!(path = %path, "loading persisted background jobs...");
        match fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str::<Vec<BackgroundJob>>(&json) {
                Ok(jobs) => {
                    let count = jobs.len();
                    let mut heap = self.jobs.lock().await;
                    for job in jobs {
                        heap.push(job);
                    }
                    info!(
                        count = count,
                        "loaded and scheduled persisted background jobs"
                    );
                    let _ = fs::remove_file(path);
                }
                Err(e) => {
                    error!(error = %e, "failed to deserialize background jobs");
                }
            },
            Err(e) => {
                error!(error = %e, "failed to read background jobs file");
            }
        }
    }

    /// Returns the number of pending jobs in the pool.
    pub async fn len(&self) -> usize {
        let heap = self.jobs.lock().await;
        heap.len()
    }

    /// Returns `true` if there are no pending jobs in the pool.
    pub async fn is_empty(&self) -> bool {
        let heap = self.jobs.lock().await;
        heap.is_empty()
    }

    /// Executes a single background job.
    async fn execute_job(&self, job: &BackgroundJob) -> Result<(), JobError> {
        info!(job_id = %job.id, job_type = ?job.job_type, "executing background job");
        let result = match &job.job_type {
            JobType::PersistEvent { run_id, event } => {
                let record = RunEventRecord::new(0, *run_id, event.clone());
                self.store
                    .runs()
                    .append_event(record)
                    .await
                    .map(|()| job.job_type.clone())
                    .map_err(|e| e.to_string())
            }
            #[cfg(feature = "queue")]
            JobType::PublishExternalEvent { event } => {
                let publisher_opt = if let Ok(lock) = self.event_publisher.read() {
                    lock.clone()
                } else {
                    None
                };
                if let Some(publisher) = publisher_opt {
                    publisher
                        .publish(event.clone())
                        .await
                        .map(|()| job.job_type.clone())
                        .map_err(|e| e.to_string())
                } else {
                    Ok(job.job_type.clone())
                }
            }
            JobType::TestJob { message } => {
                info!(message = %message, "test job executed successfully");
                Ok(job.job_type.clone())
            }
        };

        let err = match &result {
            Ok(_) => None,
            Err(e) => Some(JobError::Execution(e.clone())),
        };

        let _ = self.completed_tx.send((job.id, result));

        if let Some(e) = err { Err(e) } else { Ok(()) }
    }
}

async fn run_worker_loop(pool: Arc<BackgroundJobPool>) {
    let mut shutdown_rx = pool.shutdown_rx.clone();

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        let now = Utc::now();
        let mut sleep_duration = None;
        let mut job_to_run = None;

        {
            let mut heap = pool.jobs.lock().await;
            if let Some(top_job) = heap.peek() {
                if let Some(run_after) = top_job.conditions.run_after {
                    if run_after > now {
                        if let Ok(duration) = run_after.signed_duration_since(now).to_std() {
                            sleep_duration = Some(duration);
                        } else {
                            sleep_duration = Some(Duration::from_secs(0));
                        }
                    } else {
                        job_to_run = heap.pop();
                    }
                } else {
                    job_to_run = heap.pop();
                }
            }
        }

        if let Some(job) = job_to_run {
            let pool_clone = Arc::clone(&pool);
            tokio::spawn(async move {
                if let Err(e) = pool_clone.execute_job(&job).await {
                    error!(job_id = %job.id, error = ?e, "job execution failed");
                    if job.retries < job.max_retries {
                        let mut retried_job = job.clone();
                        retried_job.retries += 1;
                        let backoff_secs = 2 * retried_job.retries as u64;
                        retried_job.conditions.run_after =
                            Some(Utc::now() + Duration::from_secs(backoff_secs));
                        pool_clone.enqueue(retried_job).await;
                    } else {
                        error!(job_id = %job.id, "job exceeded max retries and was discarded");
                    }
                }
            });
            continue;
        }

        if let Some(duration) = sleep_duration {
            tokio::select! {
                () = tokio::time::sleep(duration) => {}
                () = pool.notify.notified() => {}
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
            }
        } else {
            tokio::select! {
                () = pool.notify.notified() => {}
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::collapsible_match,
    clippy::collapsible_if
)]
mod tests {
    use super::*;
    use crate::runtime::memory::MemoryRunStore;
    use crate::store::memory::{MemoryExecutionStore, MemorySessionStore};
    use chrono::Utc;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn make_test_store() -> Arc<RuntimeStore> {
        let sessions = MemorySessionStore::new();
        let executions = MemoryExecutionStore::new();
        let runs = MemoryRunStore::new();
        Arc::new(RuntimeStore::new(
            Box::new(sessions),
            Box::new(executions),
            Box::new(runs),
        ))
    }

    #[tokio::test]
    async fn test_job_priority_ordering() {
        let mut job_heap = std::collections::BinaryHeap::new();

        let job_low = BackgroundJob {
            id: Uuid::new_v4(),
            priority: JobPriority::Low,
            retries: 0,
            max_retries: 3,
            job_type: JobType::TestJob {
                message: "low".to_string(),
            },
            conditions: JobConditions::default(),
            created_at: Utc::now(),
        };

        let job_normal = BackgroundJob {
            id: Uuid::new_v4(),
            priority: JobPriority::Normal,
            retries: 0,
            max_retries: 3,
            job_type: JobType::TestJob {
                message: "normal".to_string(),
            },
            conditions: JobConditions::default(),
            created_at: Utc::now() + chrono::Duration::seconds(1),
        };

        let job_critical = BackgroundJob {
            id: Uuid::new_v4(),
            priority: JobPriority::Critical,
            retries: 0,
            max_retries: 3,
            job_type: JobType::TestJob {
                message: "critical".to_string(),
            },
            conditions: JobConditions::default(),
            created_at: Utc::now() + chrono::Duration::seconds(2),
        };

        let job_high = BackgroundJob {
            id: Uuid::new_v4(),
            priority: JobPriority::High,
            retries: 0,
            max_retries: 3,
            job_type: JobType::TestJob {
                message: "high".to_string(),
            },
            conditions: JobConditions::default(),
            created_at: Utc::now() + chrono::Duration::seconds(3),
        };

        job_heap.push(job_low);
        job_heap.push(job_normal);
        job_heap.push(job_critical);
        job_heap.push(job_high);

        // Should pop in priority order: Critical -> High -> Normal -> Low
        assert_eq!(job_heap.pop().unwrap().priority, JobPriority::Critical);
        assert_eq!(job_heap.pop().unwrap().priority, JobPriority::High);
        assert_eq!(job_heap.pop().unwrap().priority, JobPriority::Normal);
        assert_eq!(job_heap.pop().unwrap().priority, JobPriority::Low);
    }

    #[tokio::test]
    async fn test_job_pool_schedule_and_execution() {
        let store = make_test_store();
        let pool = BackgroundJobPool::new(store, None);

        let mut rx = pool.subscribe_completed();

        // Enqueue a simple test job
        pool.schedule(
            JobPriority::Normal,
            JobType::TestJob {
                message: "hello".to_string(),
            },
            JobConditions::default(),
        )
        .await;

        // Start worker loop
        pool.start();

        // Listen for execution completion
        let mut got_hello = false;
        for _ in 0..10 {
            if let Ok(Ok((_id, res))) =
                tokio::time::timeout(Duration::from_millis(500), rx.recv()).await
            {
                if let Ok(JobType::TestJob { message }) = res {
                    if message == "hello" {
                        got_hello = true;
                        break;
                    }
                }
            }
        }

        assert!(got_hello);

        // Shutdown cleanly
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn test_job_pool_persistence_and_recovery() {
        let store = make_test_store();
        let tmp_dir = tempdir().unwrap();
        let persist_path = tmp_dir.path().join("pending_jobs.json");
        let persist_path_str = persist_path.to_string_lossy().into_owned();

        // Create pool with persistence path and schedule a job
        let pool = BackgroundJobPool::new(store.clone(), Some(persist_path_str.clone()));

        // Push a job with high priority that won't run yet because we don't start the worker
        pool.schedule(
            JobPriority::High,
            JobType::TestJob {
                message: "persist_me".to_string(),
            },
            JobConditions::default(),
        )
        .await;

        assert_eq!(pool.len().await, 1);

        // Shutdown the pool to trigger persistence serialization
        pool.shutdown().await;

        // Verify the file exists and is populated
        assert!(persist_path.exists());

        // Create a new pool pointing to the same file
        let new_pool = BackgroundJobPool::new(store, Some(persist_path_str.clone()));

        // Recover/load persisted jobs
        new_pool.load_persisted_jobs().await;

        // Verify the job was recovered
        assert_eq!(new_pool.len().await, 1);
        let mut heap = new_pool.jobs.lock().await;
        let recovered_job = heap.pop().unwrap();
        assert_eq!(recovered_job.priority, JobPriority::High);
        if let JobType::TestJob { message } = recovered_job.job_type {
            assert_eq!(message, "persist_me");
        } else {
            panic!("Unexpected job type");
        }

        // Verify that load_persisted_jobs cleans up/deletes the persist file to prevent double-load
        assert!(!persist_path.exists());
    }
}
