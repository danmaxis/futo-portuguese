# TypoLogger patch for FUTO Keyboard

Adds local-only logging of `(typed, committed)` pairs whenever the FUTO Android Keyboard performs an autocorrect or the user manually picks a suggestion. Used to build a personal typo dataset for fine-tuning a custom language-specific FUTO model.

## Privacy

- All logs stay on the device, in the IME's per-app sandbox at
  `/sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/typo_log.jsonl`.
- No network. No analytics. No telemetry.
- `LOCALE_PREFIX` in `TypoLogger.java` filters by IME locale (`"pt"` by default — change to `"en"`, `"de"`, etc. for other languages).
- File is rotated at 50 MB → `.1` suffix.

The user pulls the log themselves via ADB when they want to use it.

## Schema (JSONL, one event per line)

```json
{"ts":"2026-05-04T12:34:56Z","typed":"voce","committed":"você","locale":"pt-BR","src":"auto_correct"}
{"ts":"2026-05-04T12:35:02Z","typed":"obigado","committed":"obrigado","locale":"pt-BR","src":"manual_pick"}
```

`src` is either `auto_correct` (keyboard substituted automatically on space/punct) or `manual_pick` (user tapped a suggestion).

## Files

| File | What it does |
|---|---|
| `TypoLogger.java` | The Java singleton — async writer thread, JSONL append, rotation. Drop into `java/src/org/futo/inputmethod/latin/utils/`. |
| `InputLogic.diff` | Two hooks in `InputLogic.java`: one inside `commitCurrentAutoCorrection` for the auto-correct path, one in `onPickSuggestionManually` for the tap path. |
| `LatinIME.diff` | Single-line `TypoLogger.init(this)` in the IME's `onCreate()`. |
| `apply_patch.sh` | Idempotent applier — copies the Java file, applies the two diffs. Run it after a fresh clone. |
| `build_and_install.sh` | `./gradlew assembleDebug` + `adb install -r`. Includes a long help message for setting up the Android command-line SDK. |

## Workflow

```bash
# 1. Clone FUTO fresh
git clone https://github.com/futo-org/android-keyboard.git futo-build
cd /path/to/this/scripts/typo_logger_patch

# 2. Apply the patch
./apply_patch.sh /path/to/futo-build

# 3. Build & install (requires JDK 17 + Android SDK; see script for setup hints)
./build_and_install.sh /path/to/futo-build

# 4. On phone: set the new (debug-signed) FUTO build as default keyboard
# 5. Use phone normally for ~1 week

# 6. Pull the log
adb pull /sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/typo_log.jsonl ./typo_log.jsonl

# 7. Hand to scripts/07_process_typo_log.py for cleanup + train/eval split
```

## Reverting

To go back to the stock FUTO build:

```bash
adb uninstall org.futo.inputmethod.latin.playstore
# Then reinstall from Play Store / FUTO website
```

The patched build uses the same package name as the stock build (we don't change the applicationId), so installing the stock APK over it will replace it cleanly. Your dictionary, learned words, and settings are stored in app data and survive the swap.

## Limitations

- **Only logs when the keyboard makes a decision.** Pure typos that the user backspaces and retypes manually (without an autocorrect or suggestion pick) are NOT captured. That's fine for our purposes — we want pairs where the keyboard already inferred "you meant X."
- **Locale filter is hardcoded to `pt`.** Edit `LOCALE_PREFIX` in `TypoLogger.java` for other languages (e.g. multi-language: leave empty to log everything).
- **Patches are against current FUTO `main`** as of 2026-05-04. If FUTO rearranges `InputLogic.java` significantly, the diff may need rebasing — line numbers in the diff are anchored on stable function names so small drift is OK.
