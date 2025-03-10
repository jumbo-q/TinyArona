import json
import numpy as np

def data_generate(json_path="../data/rawData.json",save_path='../data/data.npy')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    transcriptions = [item['transcription'] for item in data.values() if item['language'] == 'Chinese']
    
    
    transcriptions_array = np.array(transcriptions)
    
    np.save(save_path, transcriptions_array)
