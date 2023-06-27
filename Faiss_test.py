import numpy as np
import faiss
import pandas as pd

# data set initialization
data = pd.read_json('./BigData.json')
data.head()

tickedDetails = []
for i in data['ticket']:
    tickedDetails.append(i['ticketDetail'])
# print(tickedDetails)

# remove duplicates and NaN
ticket = [
    tickedDetails.replace('\n', '') for tickedDetails in list(set(tickedDetails)) if type(tickedDetails) is str
]

with open('ticketDetails.txt', 'w') as fp:
    fp.write('\n'.join(ticket))

    from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

ticket_embeddings = model.encode(ticket)
ticket_embeddings.shape
ticket_embeddings.shape[0]

with open(f'./sim_ticket/embeddings_X.npy', 'wb') as fp:
    np.save(fp, ticket_embeddings[0:256])

# saving data
split = 256
file_count = 0
for i in range(0, ticket_embeddings.shape[0], split):
    end = i + split
    if end > ticket_embeddings.shape[0] + 1:
        end = ticket_embeddings.shape[0] + 1
    file_count = '0' + str(file_count) if file_count < 0 else str(file_count)
    with open(f'./sim_ticket/embeddings_{file_count}.npy', 'wb') as fp:
        np.save(fp, ticket_embeddings[i:end, :])
    print(f"embeddings_{file_count}.npy | {i} -> {end}")
    file_count = int(file_count) + 1

d = ticket_embeddings.shape[1]

# Flat L2 Index
index = faiss.IndexFlatL2(d)
index.is_trained
index.add(ticket_embeddings)
index.ntotal
k = 4

# input ticket details
xq = model.encode(
    ["Received reports of a possible compromise of an executive's email account."])

D, I = index.search(xq, k)  # search
print(I, D)

print([f'{i}: {ticket[i]}' for i in I[0]])
