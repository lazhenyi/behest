#![allow(missing_docs, clippy::expect_used)]

fn main() {
    #[cfg(feature = "server")]
    {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        tonic_build::configure()
            .out_dir(&out_dir)
            .file_descriptor_set_path("agent_descriptor_set.bin")
            .compile_protos(
                &[
                    "src/grpc/proto/agent/v1/common.proto",
                    "src/grpc/proto/agent/v1/provider.proto",
                    "src/grpc/proto/agent/v1/session.proto",
                    "src/grpc/proto/agent/v1/run.proto",
                    "src/grpc/proto/agent/v1/tool.proto",
                    "src/grpc/proto/agent/v1/usage.proto",
                    "src/grpc/proto/agent/v1/embedding.proto",
                    "src/grpc/proto/agent/v1/artifact.proto",
                    "src/grpc/proto/agent/v1/agent.proto",
                    "src/grpc/proto/agent/v1/context.proto",
                    "src/grpc/proto/agent/v1/admin.proto",
                ],
                &["src/grpc/proto"],
            )
            .expect("failed to compile protos");
    }
}
