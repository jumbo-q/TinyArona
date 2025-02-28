import json

def generate_data(input_,output_):
    with open(input_,'r',encoding='utf-8') as f:
        lines = f.readlines()

    train_datas = []
    temp_data = ''
    for line in lines:

        if line!='\n':
            line = line.strip()
            temp_data+=(line+'\t')
        else:
            train_datas.append(temp_data)
            temp_data=''



    with open(output_,'w',encoding='utf-8') as f:
        for train_data in train_datas:
            f.write(train_data+'\n')


def get_dict(datas):
    word_count ={}
    for data in datas:
        data = data.strip().replace('\t','')
        for word in data:
            word_count.setdefault(word,0)
            word_count[word]+=1
    word2id = {"<pad>":0,"<unk>":1,"<sep>":2}

    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    id2word=list(word2id.keys())
    return word2id,id2word


if __name__ == '__main__':
    generate_data('train.txt','dataset.txt')
    with open('../dataset.txt','r',encoding='utf-8') as f:
        datas = f.readlines()
    word2id, id2word = get_dict(datas)

    dict_datas = {"word2id":word2id,"id2word":id2word}

    json.dump(dict_datas, open('../dict_datas.json', 'w', encoding='utf-8'))
