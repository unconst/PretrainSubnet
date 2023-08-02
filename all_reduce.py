


class GetParam( bt.Synapse ):
    # Compressed gradients per variable in the model.
    compressed_param: typing.Optional[typing.Dict[str, bt.Tensor]] = None

    # Sizes of compressed gradients per variable in the model.
    compressed_sizes: typing.Optional[typing.Dict[str, bt.Tensor]] = None


def all_reduce( model, subtensor, metagaph ):

    # Get all available uids.
    available = [ metagraph.axons[uid] for uid in metagraph.uids if subtensor.block - metagraph.last_update[uid] < 100 ]

    # perform all reduce.


