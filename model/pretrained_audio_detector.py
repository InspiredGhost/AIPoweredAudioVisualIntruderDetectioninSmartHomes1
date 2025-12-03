import torch
from panns_inference import AudioTagging, labels
import librosa

class PretrainedAudioEventDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = AudioTagging(checkpoint_path=None, device=device)
        self.labels = labels

    def predict(self, audio_path):
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
        waveform = torch.tensor(waveform).float().unsqueeze(0)
        # Run inference
        with torch.no_grad():
            output = self.model.inference(waveform)
        # Get top events
        events = [(self.labels[i], float(output['clipwise_output'][i])) for i in range(len(self.labels))]
        events = sorted(events, key=lambda x: x[1], reverse=True)
        return events

    def alert_events(self, audio_path, alert_classes=None, threshold=0.5):
        if alert_classes is None:
            alert_classes = ['Gunshot', 'Scream', 'Glass', 'Breaking']
        events = self.predict(audio_path)
        alerts = [e for e in events if any(cls.lower() in e[0].lower() for cls in alert_classes) and e[1] >= threshold]
        return alerts

# Example usage:
# detector = PretrainedAudioEventDetector(device='mps')
# alerts = detector.alert_events('test.wav')
# print(alerts)
