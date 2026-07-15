import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

import studio_server


def wav_payload(sample_rate: int, samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, np.clip(samples * 32767, -32768, 32767).astype(np.int16))
    return buffer.getvalue()


class StudioApiTest(unittest.TestCase):
    def test_job_manifest_is_persisted_before_any_analysis(self) -> None:
        sample_rate = 8000
        audio = np.zeros(sample_rate, dtype=np.float32)
        original_output_root = studio_server.OUTPUT_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            studio_server.OUTPUT_ROOT = Path(tmp)
            try:
                client = TestClient(studio_server.app)
                response = client.post(
                    "/api/jobs",
                    files={"heartbeat": ("heartbeat.wav", wav_payload(sample_rate, audio), "audio/wav")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                manifest_path = Path(tmp) / payload["job_id"] / "manifest.json"
                self.assertTrue(manifest_path.is_file())
                self.assertEqual(payload["manifest"]["schema_version"], 2)
                self.assertEqual(payload["stages"]["heartbeat"]["status"], "pending")
                self.assertIn("sha256", payload["manifest"]["inputs"]["heartbeat"])
            finally:
                studio_server.OUTPUT_ROOT = original_output_root

    def test_song_pipeline_runs_independently_and_preserves_other_stages(self) -> None:
        sample_rate = 8000
        pulse = np.hanning(120).astype(np.float32)
        song = np.zeros(sample_rate * 4, dtype=np.float32)
        for time_seconds in np.arange(0.25, 4.0, 0.5):
            start = int(time_seconds * sample_rate)
            song[start : start + len(pulse)] += pulse

        original_output_root = studio_server.OUTPUT_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            studio_server.OUTPUT_ROOT = Path(tmp)
            try:
                client = TestClient(studio_server.app)
                created = client.post(
                    "/api/jobs",
                    files={"song": ("song.wav", wav_payload(sample_rate, song), "audio/wav")},
                ).json()
                job_id = created["job_id"]
                response = client.post(
                    f"/api/jobs/{job_id}/song/analyze",
                    json={"bpm_override": 120.0, "first_beat_override": 0.25, "beats_per_bar": 4},
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["stages"]["song"]["status"], "ok")
                self.assertEqual(payload["stages"]["heartbeat"]["status"], "pending")
                tempo_map = payload["stages"]["song"]["outputs"]["song_tempo_map_csv"]
                self.assertTrue((Path(tmp) / job_id / tempo_map).is_file())
                blocked = client.post(f"/api/jobs/{job_id}/render", json={})
                self.assertEqual(blocked.status_code, 409)
            finally:
                studio_server.OUTPUT_ROOT = original_output_root

    def test_process_render_and_media_contract(self) -> None:
        sample_rate = 8000
        pulse = np.hanning(160).astype(np.float32)
        heartbeat = np.zeros(sample_rate * 7, dtype=np.float32)
        for second in range(7):
            heartbeat[second * sample_rate : second * sample_rate + len(pulse)] += pulse
        song = np.zeros(sample_rate * 4, dtype=np.float32)
        for time_seconds in np.arange(0.25, 4.0, 0.5):
            start = int(time_seconds * sample_rate)
            song[start : start + len(pulse)] += pulse * 0.25

        original_output_root = studio_server.OUTPUT_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            studio_server.OUTPUT_ROOT = Path(tmp)
            try:
                client = TestClient(studio_server.app)
                response = client.post(
                    "/api/process",
                    files={
                        "heartbeat": ("heartbeat.wav", wav_payload(sample_rate, heartbeat), "audio/wav"),
                        "song": ("song.wav", wav_payload(sample_rate, song), "audio/wav"),
                    },
                    data={
                        "bpm": "120",
                        "first_beat": "0.25",
                        "beats_per_bar": "4",
                        "loop_beats": "4",
                        "heartbeat_gain_db": "-15",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["analysis"]["estimated_bpm"], 120.0)
                self.assertEqual({track["id"] for track in payload["tracks"]}, {"song", "heartbeat"})
                media_response = client.get(payload["tracks"][1]["url"])
                self.assertEqual(media_response.status_code, 200)
                self.assertGreater(len(media_response.content), 100)

                render = client.post(
                    f"/api/jobs/{payload['job_id']}/render",
                    json={"bpm": 100.0, "first_beat_seconds": 0.5, "heartbeat_gain_db": -12.0},
                )
                self.assertEqual(render.status_code, 200, render.text)
                rendered = render.json()
                self.assertEqual(rendered["revision"], 2)
                self.assertAlmostEqual(rendered["mix_report"]["target_loop_duration_seconds"], 2.4, places=6)
                self.assertEqual(client.get(rendered["final_mix_wav_url"]).status_code, 200)
            finally:
                studio_server.OUTPUT_ROOT = original_output_root


if __name__ == "__main__":
    unittest.main()
