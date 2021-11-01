import os

from data.PlantVillageDataset import PlantVillage

def generateDataset(plant, transform):
    # Plant Village dataset
    if not os.path.exists(f'./data/plantvillage/{plant}/train/data.csv'):
        PlantVillage.create_csv(f'./data/plantvillage/{plant}/train')

    if not os.path.exists(f'./data/plantvillage/{plant}/val/data.csv'):
        PlantVillage.create_csv(f'./data/plantvillage/{plant}/val')

    if not os.path.exists(f'./data/plantvillage/{plant}/test/data.csv'):
        PlantVillage.create_csv(f'./data/plantvillage/{plant}/test')

    plantVillageTrain = PlantVillage(csv_file=f'./data/plantvillage/{plant}/train/data.csv',
                                     root_dir=f'./data/plantvillage/{plant}/train',
                                     transform=transform)

    plantVillageVal = PlantVillage(csv_file=f'./data/plantvillage/{plant}/val/data.csv',
                                   root_dir=f'./data/plantvillage/{plant}/val',
                                   transform=transform)

    plantVillageTest = PlantVillage(csv_file=f'./data/plantvillage/{plant}/test/data.csv',
                                    root_dir=f'./data/plantvillage/{plant}/test',
                                    transform=transform)

    return plantVillageTrain, plantVillageVal, plantVillageTest
