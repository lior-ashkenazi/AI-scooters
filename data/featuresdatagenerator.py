from programio.abstractio import AbstractIO


class FeaturesData:
    """
    this class constitutes the interface of the agent to the features
    """


class FeaturesDataGenerator:
    """
    this class is similar in terms of flow to TrafficGenerator.
    it should interact with the user in order to get features for the user.
    we assume that we have a file that contains data that we can use in order to
     create features.
    this class should operate similar to TrafficGenerator, to get requests from the user
    and return the requested features (that's why it has io attribute)
    we should decide what is the type of the features to return  (implementation of
     FeaturesData)
    """
    def __init__(self, io: AbstractIO):
        self.io = io

    def generate_features_data(self) -> FeaturesData:
        pass




