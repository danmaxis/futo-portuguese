#!/usr/bin/env bash
# Build the patched FUTO APK and install over an ADB-connected phone.
#
# Prereqs (on the build host):
#   - JDK 17 (`pacman -S jdk17-openjdk` on Manjaro)
#   - Android SDK with build-tools 34+ and platform-tools (`pacman -S android-tools`,
#     plus install Android SDK via sdkmanager — see SDK_MANAGER_HINT below)
#   - ANDROID_HOME / ANDROID_SDK_ROOT pointing at the SDK
#   - The FUTO repo with the TypoLogger patch already applied (see apply_patch.sh)
#   - The phone paired via wireless ADB and connected.
#
# Usage:
#   ./build_and_install.sh /path/to/futo-checkout
#
# The output APK lands at:
#   <repo>/build/outputs/apk/playstore/debug/java-playstore-debug.apk
# (path may differ depending on flavour — script auto-detects).

set -euo pipefail

REPO="${1:?usage: $0 <futo-repo-path>}"

cd "$REPO"

if [ -z "${ANDROID_HOME:-}${ANDROID_SDK_ROOT:-}" ]; then
    cat <<'EOF' >&2
ERROR: ANDROID_HOME / ANDROID_SDK_ROOT not set.

Quickest setup on Manjaro/Arch (no Android Studio needed):

    sudo pacman -S jdk17-openjdk android-tools
    sudo archlinux-java set java-17-openjdk
    # Download command-line tools (small):
    mkdir -p ~/Android/cmdline-tools && cd ~/Android/cmdline-tools
    curl -OL https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
    unzip commandlinetools-linux-*.zip && mv cmdline-tools latest
    export ANDROID_HOME=~/Android
    export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools
    yes | sdkmanager --licenses
    sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;26.3.11579264"

Then re-run this script.
EOF
    exit 1
fi

echo "==> Building debug APK (gradle assembleDebug)"
./gradlew assembleDebug

echo
echo "==> Locating APK"
APK="$(find build/outputs/apk -name '*-debug.apk' | head -1)"
if [ -z "$APK" ]; then
    APK="$(find . -name '*-debug.apk' -not -path './build-tools/*' | head -1)"
fi
if [ -z "$APK" ] || [ ! -f "$APK" ]; then
    echo "ERROR: no debug APK produced." >&2
    exit 2
fi
ls -lh "$APK"

echo
echo "==> Verifying ADB device"
adb devices | tee /dev/stderr | grep -q 'device$' || {
    echo "ERROR: no ADB device. Reconnect via wireless ADB." >&2
    exit 3
}

echo
echo "==> Installing (replacing any existing version)"
adb install -r "$APK"

echo
echo "==> Done. Set FUTO Keyboard as your default IME and start typing."
echo "    Pull logs anytime via:"
echo "    adb pull /sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/typo_log.jsonl"
