from tqdm import tqdm

from domain_adapt.data.office31 import Office31

if __name__ == '__main__':
    root_dir = '/home/mchobanyan/data/research/transfer/office'
    amazon_data = Office31(root_dir, 'amazon', source=False)

    num_runs = 20
    for run in tqdm(range(num_runs)):
        # prepare the models (feature extractor, label classifier, domain classifier)
        pass

    print(len(amazon_data))
