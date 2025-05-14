{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    conda
    sox
  ];

  shellHook = ''
    conda-shell
  '';
}
