import av 
import torch
import torchvision.transform as T 
from typing import Generator, Tuple


class VideoStreamLoader:
    def __init__(self, video_path: str, target_fps: str, img_size: int):
        self.video_path = video_path
        self.target_fps = target_fps

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def stream_frames(self) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generator returning (frame_tensor [C, H, W], timestamp_seconds).
        """
        container = av.open(self.video_path)
        stream = container.streams.video[0]

        # Time interval between two target frames
        time_base = float(stream.time_base)
        sample_interval = 1.0 / self.target_fps
        next_sample_time = 0.0

        for frame in container.decode(stream):
            # Exact timestamp of the frame in seconds
            current_time = frame.pts * time_base if frame.pts is not None else 0.0

            if current_time >= next_sample_time:
                # Conversion PyAV frame -> numpy array RGB -> Tensor
                img = frame.to_ndarray(format="rgb24")
                img_tensor = torch.from_numpy(img).permute(2, 0, 1) # [H, W, C] -> [C, H, W]
                
                processed_frame = self.transform(img_tensor)
                yield processed_frame, current_time
                
                next_sample_time += sample_interval

        container.close()