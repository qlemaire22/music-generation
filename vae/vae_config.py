# General values
BATCH_SIZE = 50
ORIGINAL_DIM = 512 * 6  # 3072
LATENT_DIM = 4
EPSILON_STD = 1.0
NUMBER_EPOCHS = 100
NB_VECT = BATCH_SIZE * 22  # 704

# Dimension values for basic VAE
INTER_DIM = 50

# Dimension values for the deep VAE
INTER_DIM1 = int(ORIGINAL_DIM / 4)  # 768
INTER_DIM2 = int(INTER_DIM1 / 4)  # 192
INTER_DIM3 = int(INTER_DIM2 / 4)  # 48
