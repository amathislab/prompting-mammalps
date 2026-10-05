from salma.utils.misc import Mode
import torch.distributed as dist
import math


class TestSequentialVideoLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        cropping_mode: str | None = None,
        collate_fn=None,
        rank: int | None = None,
        world_size: int | None = None,
        logger = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cropping_mode = cropping_mode
        self.collate_fn = collate_fn

        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1

        self.rank = rank
        self.world_size = world_size

        #self.view_id = rank % 2            # 0 = left, 1 = right
        #self.view_rank = rank // 2
        #self.view_world_size = math.ceil(world_size / 2)

        all_indices = list(range(len(dataset)))
        #self.local_video_indices = all_indices[self.view_rank :: self.view_world_size]
        self.local_video_indices = all_indices[self.rank :: self.world_size]
        self.num_videos = len(self.local_video_indices)

        self.logger = logger

    def __len__(self):
        # approximate number of iterations
        return math.ceil(self.num_videos / self.batch_size)

    def __iter__(self):
        self.next_video_ptr = 0

        self.slots = []
        for slot_id in range(self.batch_size):
            if self.next_video_ptr < self.num_videos:
                self.logger.debug(f"Starting to process {self.local_video_indices[self.next_video_ptr]}")
                self.slots.append(
                    {
                        "slot_id": slot_id,
                        "video_index": self.local_video_indices[self.next_video_ptr],
                        "active": True,
                    }
                )
                self.next_video_ptr += 1
            else:
                self.slots.append(
                    {
                        "slot_id": slot_id,
                        "video_index": None,
                        "active": False,
                    }
                )

        return self

    def __next__(self):
        active_slots = [s for s in self.slots if s["active"]]

        if not active_slots:
            # No more active slots: we went through all the videos
            raise StopIteration

        batch = []
        batch_meta = {
            "slot_id": [],
            "video_index": [],
        }

        for slot in active_slots:
            idx = slot["video_index"]

            # Get the next frames from the current video
            sample = self.dataset.__getitem__(
                idx,
                cropping_mode = self.cropping_mode
            )

            if sample is None:
                self.logger.debug(f"Finished processing {slot['video_index']}")
                # video exhausted -> replace with next local video
                if self.next_video_ptr < self.num_videos:
                    slot["video_index"] = self.local_video_indices[self.next_video_ptr]
                    self.next_video_ptr += 1
                    self.logger.debug(f"Starting to process {slot['video_index']}")

                    sample = self.dataset.__getitem__(
                        slot["video_index"],
                    )
                else:
                    # No more videos in the dataset, we set as inactive slot
                    slot["active"] = False
                    continue

            batch.append(sample)
            batch_meta["slot_id"].append(slot["slot_id"])
            batch_meta["video_index"].append(slot["video_index"])


        if not batch:
            raise StopIteration

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        return {
            "batch": batch,
            "meta": batch_meta,
        }

    
def build_test_loader(
    dataset,
    batch_size: int,
    cropping_mode: str | None = None,
    collate_fn=None,
    rank: int | None = None,
    world_size: int | None = None,
    logger = None
):
    """
    Build a sequential distributed test-time video loader.
    Dataset mode must be TEST.
    """

    if dataset.mode != Mode.TEST:
        raise ValueError(
            f"build_test_loader called with dataset mode {dataset.mode}"
        )


    return TestSequentialVideoLoader(
        dataset=dataset,
        batch_size=batch_size,
        cropping_mode = cropping_mode,
        collate_fn=collate_fn,
        rank=rank,
        world_size=world_size,
        logger=logger
    )
