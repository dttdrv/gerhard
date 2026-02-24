import json  
nb=json.load(open('notebooks/asnn_goose_colab_v15.ipynb'))  
keywords=('Phase','MI','CKA','gating','dossier')  
for i,cell in enumerate(nb['cells']):  
    src=''.join(cell.get('source', []))  
    if any(k in src for k in keywords):  
        print('Cell', i, 'Type', cell['cell_type'])  
        print(src)  
        print('-'*40)  
