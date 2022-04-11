import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, apply_scaler

if __name__=="__main__":
    decoder_path = '../output/20220405-15:14:02_decoder.keras'
    encoder_path = '../output/20220405-15:14:02_encoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing = get_scaled_data(put_nans_back=True)
    m_datasets = 1
    for i in range(m_datasets):
        index_changes = model.impute_multiple(data_corrupt=data_missing, max_iter=1000)
        plt.hist(index_changes, range=[0,134], bins=133)
        plt.show()
        plt.savefig('1000_iterations_n_changes_per_index')
