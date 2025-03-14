from math import isclose


class EarlyStopper:
    """
    Simple early stopper that compares the minimum validation loss with a minimum loss delta `min_delta`.
    Stops after `patience` number of steps with no improvement of at least `min_delta`, i.e. Stop after `patience` steps of no decrease by `min`delta`.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):  # model is improving
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):  # no significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
