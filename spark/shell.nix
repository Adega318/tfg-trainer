{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "pipzone";

  targetPkgs = pkgs:
    (with pkgs; [ python3 python3Packages.pip python3Packages.virtualenv ]);
}).env
