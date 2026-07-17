{
  description = "Garasu (硝子) — GPU rendering engine: wgpu pipeline, text rendering, shader system, and winit integration";

  # substrate.rust.library dispatches over Cargo.gen.lock (the slim gen delta,
  # reconstructed to the full BuildSpec in pure Nix) — no crate2nix, no Cargo.nix.
  inputs.substrate.url = "github:pleme-io/substrate";

  outputs = { substrate, ... }: substrate.rust.library {
    src = ./.;
  };
}
