# integration.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTrackingWrapper:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)
        self.alerted_ids = set()

    def update(self, detections, frame):
        print(detections)
        input_dets = []
        for det in detections:
            box = det.get("box")
            if isinstance(box, float) or box is None:
                print(f"[ERROR] Invalid box type (float or None): {box}")
                continue

            if not isinstance(box, (list, tuple)) or len(box) != 4:
                print(f"[WARNING] Invalid bbox format: {box}")
                continue

            try:
                x1, y1, x2, y2 = map(float, box)
                conf = float(det["confidence"])
                label = str(det["label"])
                input_dets.append([x1, y1, x2, y2, conf, label])
            except Exception as e:
                print(f"[EXCEPTION] Failed to parse detection: {det} | Error: {e}")
                continue

        return self.tracker.update_tracks(input_dets, frame=frame)

    def should_alert(self, track_id):
        if track_id not in self.alerted_ids:
            self.alerted_ids.add(track_id)
            return True
        return False