{
  description = "UMA Assignments dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.texlive.combined.scheme-full

            pkgs.cmake
            pkgs.pkg-config
            pkgs.stdenv.cc.cc.lib
            pkgs.gfortran
            pkgs.openblas

            # required for xgboost on mac
            pkgs.llvmPackages.openmp
          ];

          
          buildInputs = [
            pkgs.arrow-cpp
          ];

          shellHook = ''
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            cd code
            if [ ! -d .venv ] || [ pyproject.toml -nt .venv ] || [ uv.lock -nt .venv ]; then
              touch .venv
            fi
            source .venv/bin/activate
            cd ..
          '';
        };
      }
    );
}
