/**
 * HealthShield — Human Voice Companion (Web Speech API)
 * Reads out the daily health narrative and clinical advice in a warm,
 * empathetic voice for accessible, hands-free healthcare check-ins.
 */

class HealthSpeechNarrator {
    constructor() {
        this.synth = window.speechSynthesis || null;
        this.isSpeaking = false;
        this.currentUtterance = null;
    }

    isSupported() {
        return !!this.synth;
    }

    stop() {
        if (this.synth) {
            this.synth.cancel();
            this.isSpeaking = false;
            this.updateUiState(false);
        }
    }

    speak(text, onComplete = null) {
        if (!this.synth) {
            console.warn("Web Speech API not supported on this browser.");
            return;
        }

        this.stop();

        const cleanText = text
            .replace(/[#*_`]/g, "")
            .replace(/BPM/g, "beats per minute")
            .replace(/SpO2/g, "oxygen saturation")
            .replace(/mmHg/g, "millimeters of mercury")
            .replace(/hrs/g, "hours");

        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 0.94; // slightly slower, calm bedside manner
        utterance.pitch = 1.0;

        // Try to select a warm, natural English voice
        const voices = this.synth.getVoices();
        const preferred = voices.find(v => 
            v.lang.startsWith("en") && (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Samantha") || v.name.includes("Arthur"))
        ) || voices.find(v => v.lang.startsWith("en"));

        if (preferred) {
            utterance.voice = preferred;
        }

        utterance.onstart = () => {
            this.isSpeaking = true;
            this.updateUiState(true);
        };

        utterance.onend = () => {
            this.isSpeaking = false;
            this.updateUiState(false);
            if (onComplete) onComplete();
        };

        utterance.onerror = (e) => {
            console.warn("Speech synthesis error", e);
            this.isSpeaking = false;
            this.updateUiState(false);
        };

        this.currentUtterance = utterance;
        this.synth.speak(utterance);
    }

    updateUiState(active) {
        const btn = document.getElementById("audio-narrator-btn");
        const waves = document.getElementById("audio-soundwaves");
        if (btn) {
            btn.classList.toggle("speaking", active);
            const textSpan = btn.querySelector(".narrator-text");
            if (textSpan) {
                textSpan.textContent = active ? "Pause Voice Summary" : "Listen to Daily Brief";
            }
        }
        if (waves) {
            waves.style.display = active ? "inline-flex" : "none";
        }
    }
}

window.HealthNarrator = new HealthSpeechNarrator();
