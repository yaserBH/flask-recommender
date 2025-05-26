import os
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import os, io, boto3, torch

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv("https://storage.c2.liara.space"),
    aws_access_key_id=os.getenv("7c5kqlsc3914si9c"),
    aws_secret_access_key=os.getenv("014cc8cc-8933-4314-9e10-b34dc75f2637"),
)
def fetch_pt(key):
    resp = s3.get_object(Bucket="flask-bucket-recommend", Key=key)
    return torch.load(io.BytesIO(resp['Body'].read()))

bag_vectors  = fetch_pt("bag_features.pt")
user_vectors = fetch_pt("user_vectors.pt")

print(f"→ {len(bag_vectors)} bags, {len(user_vectors)} users loaded")

app = Flask(__name__)

def recommend_bags_for_user(user_id, top_k=5):
    if user_id not in user_vectors:
        return []

    skus      = list(bag_vectors.keys())      
    feats     = torch.stack([bag_vectors[s] for s in skus]) 
    user_emb  = user_vectors[user_id].unsqueeze(0)           

    feats_norm = F.normalize(feats, dim=1)
    user_norm  = F.normalize(user_emb, dim=1)
    sims       = torch.mm(user_norm, feats_norm.T).squeeze(0) 

    topk_idxs = sims.topk(k=top_k).indices.tolist()
    return [skus[i] for i in topk_idxs]

@app.route('/recommend', methods=['GET'])
def recommend_endpoint():
    raw = request.args.get('user_id')
    if raw is None:
        return jsonify({"error": "user_id is required"}), 400

    user_id = raw  

    recs = recommend_bags_for_user(user_id, top_k=5)
    return jsonify({
        "user_id":       user_id,
        "recommendations": recs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
