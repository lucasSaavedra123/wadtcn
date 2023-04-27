MINUTES_PER_EPOCH = 1
INITIAL_EPOCHS = 5
STEP = 5

def estimate_succesive_halving(number_of_combinations):
    if number_of_combinations == 0:
        return 0
    elif number_of_combinations == 1:
        return 0
    else:
        return estimate_succesive_halving(number_of_combinations//2) + STEP * number_of_combinations * MINUTES_PER_EPOCH

def estimate_hyperparameter_search(number_of_combinations):
    return number_of_combinations * MINUTES_PER_EPOCH * INITIAL_EPOCHS + estimate_succesive_halving(number_of_combinations)

print(estimate_hyperparameter_search(196))