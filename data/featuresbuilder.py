from programio.abstractio import AbstractIO


class FeaturesBuilder:
    """
    this class is similar in terms of flow to DataGenerator.
    it should interact with the user in order to get features for the user.
    we assume that we have a file that contains data that we can use in order to
     create features.
    this class should operate similar to DataGenerator, to get requests from the user
    and return the requested features (that's why it has io attribute)
    we should decide what is the type (simple lists? new class?) of the features
     to return
    """
    def __init__(self, io: AbstractIO):
        self.io = io


