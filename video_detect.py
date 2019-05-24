import av
import numpy as np
import tqdm
from datetime import timedelta
from time import time

'''
input = av.open("D:/Downloads/test.mp4")

duration = {
    'container': float(input.duration / av.time_base),
    'video':     float(input.streams.video[0].duration * input.streams.video[0].time_base),
    'audio':     float(input.streams.audio[0].duration * input.streams.audio[0].time_base),
}
for k, v in duration.items():
    print("duration by {}: {}".format(k, timedelta(seconds=v)))

output = av.open("W:/output.mp4", 'w')

c_video = 0
c_audio = 0
t1 = time()

stream_map = {} # maps streams in input container to their equivalent in output container
with tqdm.tqdm(total=input.streams.video[0].frames) as progress_bar:
    for packet in input.demux():

        if packet.dts is None: continue
        if packet.pts < packet.dts: continue # error in input file that sometimes occurs.

        if packet.stream not in stream_map.keys():
            if packet.stream.type == 'video':
                out_stream = output.add_stream('h264')
                out_stream.codec_context.bit_rate = packet.stream.codec_context.bit_rate
                out_stream.codec_context.width = packet.stream.codec_context.width
                out_stream.codec_context.height = packet.stream.codec_context.height
                out_stream.codec_context.framerate = packet.stream.codec_context.framerate
                out_stream.codec_context.rate = packet.stream.codec_context.rate
            else:
                out_stream = output.add_stream(template=packet.stream)

            stream_map[packet.stream] = out_stream



        out_stream = stream_map[packet.stream]

        if packet.stream.type != 'video':
            packet.stream = out_stream
            output.mux(packet)
        else:
            for frame in packet.decode():
                rgb = frame.to_rgb().to_ndarray()
                rgb[rgb != 0] //= 2
                frame = av.VideoFrame.from_ndarray(rgb)
                for out_packet in out_stream.encode(frame):
                    output.mux(out_packet)
                progress_bar.update(1)

    for video_stream in [stream for stream in stream_map.values() if stream.type == 'video']:
        for flush_packet in video_stream.encode():
            output.mux(flush_packet)

    output.close()

t2 = time()
print("interleaved streams time:", timedelta(seconds=t2-t1), c_video, c_audio)
'''

def copy_non_video_streams(output, input):
    input.seek(0, format='frame')
    output.seek(0, format='frame')

    stream_map = {}
    for packet in input.demux():
        if packet.dts is None: continue
        if packet.pts < packet.dts: continue
        if packet.stream.type == 'video': continue
        
        if packet.stream not in stream_map.keys():
            stream_map[packet.stream] = output.add_stream(template=stream)

        out_stream = stream_map[packet.stream]
        packet.stream = out_stream
        output.mux(packet)

import torch
from torch.utils.data import Dataset, DataLoader
from utils.datasets import letterbox
import matplotlib
    

class VideoStreamDataset(Dataset):
    def __init__(self, video_stream, image_size=416):
        self.image_size = image_size
        video_stream.seek(0, whence='frame')
        self.stream = video_stream
        self.count = video_stream.frames
        self.current_index = 0
        self.current_frame: torch.Tensor
        self.hw: Tuple[int, int]
    
    def __len__(self):
        return self.count   
    
    def __getitem__(self, index):
        if index - self.current_frame > 32 or index - self.current_frame < 0:
            self.stream.seek(index, whence='frame')

        for packet in self.stream.demux():
            if packet.dts is None: continue
            if packet.pts < packet.dts: continue

            for frame in packet.decode():
                if frame.index > index:
                    raise StopIteration
                if frame.index == index:
                    pixels = frame.to_rgb().to_ndarray()
                    pixels, _, _, _ = letterbox(pixels, new_shape=self.image_size)
                    self.current_frame = torch.from_numpy(pixels)
                    self.hw = (frame.height, frame.width)



class VideoDetectionDataset(Dataset):  # for training/testing
    def __init__(self, video_stream, img_size=416, batch_size=16, rect=True):
        video_stream.seek(0, whence='frame')
        self.stream = video_stream
        self.count = video_stream.frames
        self.current_index = 0
        self.hw: Tuple[int, int]

        n = self.count
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, "No frames found in stream."

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.rect = rect

    '''
        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            from PIL import Image

            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            if os.path.exists(sp):  # read existing shapefile
                with open(sp, 'r') as f:
                    s = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                assert len(s) == n, 'Shapefile out of sync, please delete %s and rerun' % sp
            else:  # no shapefile, so read shape using PIL and write shapefile for next time (faster)
                s = np.array([Image.open(f).size for f in tqdm(self.img_files, desc='Reading image shapes')])
                np.savetxt(sp, s, fmt='%g')

            # Sort by aspect ratio
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            ar = ar[i]
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32
    '''

    def __len__(self):
        return self.count

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):

        if index - self.current_index > 32 or index - self.current_index < 0:
            self.stream.seek(index, whence='frame')

        image = None
        for packet in self.stream.container.demux():
            if packet.dts is None: continue
            if packet.pts < packet.dts: continue
            if packet.stream is not self.stream: continue

            for frame in packet.decode():
                if frame.index > index:
                    raise StopIteration
                if frame.index == index:
                    self.current_index = index
                    image = frame.to_rgb().to_ndarray()
                    break
            if image is not None: break
        assert image is not None, "No frame at index %d found." % index

        # Letterbox
        h, w, _ = image.shape
        if self.rect:
            #shape = self.batch_shapes[self.batch[index]]
            wh_ratio = w / h
            if w > h:
                shape = (self.img_size / wh_ratio, self.img_size)
                shape = (int(np.floor(shape[0] / 32) * 32 + 32), shape[1])
            else:
                shape = (self.img_size, self.img_size * wh_ratio)
                shape = (shape[0], int(np.floor(shape[1] / 32) * 32 + 32))
            image, ratio, padw, padh = letterbox(image, new_shape=shape, mode='rect')
        else:
            shape = self.img_size
            image, ratio, padw, padh = letterbox(image, new_shape=shape, mode='square')

        # Normalize
        image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(image), (h, w)

    @staticmethod
    def collate_fn(batch):
        img, hw = list(zip(*batch))  # transposed
        stacked = torch.stack(img, 0)
        return stacked, hw


input = av.open("D:/Downloads/test.mp4")
video_stream = input.streams.video[0]

dataset = VideoDetectionDataset(video_stream)

dataloader = DataLoader(dataset,
                        batch_size=16,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)

for i, (frame, hw) in enumerate(dataloader):
    sbs = frame.reshape((frame.shape[0] * frame.shape[1], frame.shape[2], frame.shape[3]))
