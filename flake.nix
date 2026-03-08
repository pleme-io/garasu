{
  description = "Garasu (硝子) — GPU rendering engine: wgpu pipeline, text, shaders, winit";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    substrate = {
      url = "github:pleme-io/substrate";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crate2nix.url = "github:nix-community/crate2nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      substrate,
      crate2nix,
      ...
    }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };
      substrateLib = substrate.libFor.${system} nixpkgs;
      rustLibrary = import "${substrate}/lib/rust-library.nix" {
        inherit system nixpkgs;
        nixLib = substrate;
        inherit crate2nix;
      };
      lib = rustLibrary {
        name = "garasu";
        src = ./.;
      };
    in
    {
      inherit (lib) packages devShells apps;

      overlays.default = final: prev: {
        garasu = self.packages.${final.system}.default;
      };

      formatter.${system} = pkgs.nixfmt-tree;
    };
}
