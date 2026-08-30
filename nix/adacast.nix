{
  lib,
  stdenvNoCC,
  callPackage,
  bun,
  sysctl,
  makeBinaryWrapper,
  models-dev,
  ripgrep,
  installShellFiles,
  versionCheckHook,
  writableTmpDirAsHomeHook,
  node_modules ? callPackage ./node-modules.nix { },
}:
stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "adacast";
  inherit (node_modules) version src;
  inherit node_modules;

  nativeBuildInputs = [
    bun
    installShellFiles
    makeBinaryWrapper
    models-dev
    writableTmpDirAsHomeHook
  ];

  configurePhase = ''
    runHook preConfigure

    cp -R ${finalAttrs.node_modules}/. .

    runHook postConfigure
  '';

  env.MODELS_DEV_API_JSON = "${models-dev}/dist/_api.json";
  env.ADACAST_DISABLE_MODELS_FETCH = true;
  env.ADACAST_VERSION = finalAttrs.version;
  env.ADACAST_CHANNEL = "local";

  buildPhase = ''
    runHook preBuild

    cd ./packages/adacast
    bun --bun ./script/build.ts --single --skip-install
    bun --bun ./script/schema.ts schema.json

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    install -Dm755 dist/adacast-*/bin/adacast $out/bin/adacast
    install -Dm644 schema.json $out/share/adacast/schema.json
    cp -R dist/adacast-*/bin/app $out/share/adacast/app
    cp -R dist/adacast-*/bin/python $out/share/adacast/python

    wrapProgram $out/bin/adacast \
      --prefix PATH : ${
        lib.makeBinPath (
          [
            ripgrep
          ]
          # bun runs sysctl to detect if dunning on rosetta2
          ++ lib.optional stdenvNoCC.hostPlatform.isDarwin sysctl
        )
      }

    runHook postInstall
  '';

  postInstall = lib.optionalString (stdenvNoCC.buildPlatform.canExecute stdenvNoCC.hostPlatform) ''
    # trick yargs into also generating zsh completions
    installShellCompletion --cmd adacast \
      --bash <($out/bin/adacast completion) \
      --zsh <(SHELL=/bin/zsh $out/bin/adacast completion)
  '';

  nativeInstallCheckInputs = [
    versionCheckHook
    writableTmpDirAsHomeHook
  ];
  doInstallCheck = true;
  versionCheckKeepEnvironment = [ "HOME" "ADACAST_DISABLE_MODELS_FETCH" ];
  versionCheckProgramArg = "--version";

  passthru = {
    jsonschema = "${placeholder "out"}/share/adacast/schema.json";
  };

  meta = {
    description = "AdaCast TUI agent CLI";
    homepage = "https://example.invalid/AdaCast/";
    license = lib.licenses.mit;
    mainProgram = "adacast";
    inherit (node_modules.meta) platforms;
  };
})
