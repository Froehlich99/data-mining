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
        apps.dataset-stats = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-dataset-stats" ''
            set -euo pipefail
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/dataset_stats.ipynb \
              notebooks/dataset_stats.ipynb \
              -p data_path data/features.csv
          '');
        };

        apps.feature-outliers = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-feature-outliers" ''
            set -euo pipefail
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/feature_outliers.ipynb \
              notebooks/feature_outliers.ipynb \
              -p data_path data/features.csv \
              -p n_extremes 3
          '');
        };

        apps.clustering-mebeauty = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-clustering-mebeauty" ''
            set -euo pipefail
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            mkdir -p data/plots/clustering_mebeauty
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/clustering_mebeauty.ipynb \
              notebooks/clustering_mebeauty.ipynb \
              -p data_path data/features.csv \
              -p output_dir data/plots/clustering_mebeauty
          '');
        };

        apps.clustering = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-clustering" ''
            set -euo pipefail
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            mkdir -p data/plots/clustering
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/clustering.ipynb \
              notebooks/clustering.ipynb \
              -p data_path data/features.csv \
              -p output_dir data/plots/clustering
          '');
        };

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

        apps.papermill = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-papermill" ''
            set -euo pipefail
            cd "$(git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/feature_distributions.ipynb \
              notebooks/feature_distributions.ipynb \
              "$@"
            echo "Executed: code/notebooks/feature_distributions.ipynb"
          '');
        };

        apps.feature-pruning = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-feature-pruning" ''
            set -euo pipefail
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)/code"
            export UV_PYTHON=${python}/bin/python3
            export DYLD_LIBRARY_PATH=${pkgs.llvmPackages.openmp}/lib''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            mkdir -p data/plots/feature_pruning
            ${pkgs.uv}/bin/uv run --extra notebooks papermill \
              notebooks/feature_pruning.ipynb \
              notebooks/feature_pruning.ipynb \
              -p data_path data/features.csv \
              -p output_dir data/plots/feature_pruning
          '');
        };
      }
    );
}
