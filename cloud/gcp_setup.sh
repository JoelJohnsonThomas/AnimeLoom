#!/bin/bash
# =========================================================================
# Google Cloud VM setup for AnimeLoom
# Uses $300 free credits → T4 GPU at ~$0.35/hr → 850+ hours
# =========================================================================
set -e

# Configuration
INSTANCE_NAME="${GCP_INSTANCE_NAME:-animeloom-vm}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="100GB"
PROJECT_REPO="${PROJECT_REPO:-https://github.com/yourname/anime-character-engine}"

echo "=== AnimeLoom GCP Setup ==="
echo "Instance:  $INSTANCE_NAME"
echo "Zone:      $ZONE"
echo "GPU:       $GPU_TYPE x $GPU_COUNT"
echo ""

# Create instance with GPU
echo "Creating VM..."
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="$DISK_SIZE" \
    --metadata=install-nvidia-driver=True

echo "Waiting for instance to start..."
sleep 30

# Setup inside the VM
echo "Configuring VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    set -e

    # Clone repository
    git clone $PROJECT_REPO ~/animeloom || (cd ~/animeloom && git pull)
    cd ~/animeloom

    # Setup environment
    chmod +x setup.sh
    ./setup.sh

    # Start the API server in background
    source venv/bin/activate
    nohup uvicorn api.app:app --host 0.0.0.0 --port 8080 > api.log 2>&1 &

    echo 'Setup complete! API running on port 8080'
"

# Open firewall for API
gcloud compute firewall-rules create allow-animeloom-api \
    --allow=tcp:8080 \
    --target-tags=animeloom \
    --description="Allow AnimeLoom API traffic" 2>/dev/null || true

# Tag the instance
gcloud compute instances add-tags "$INSTANCE_NAME" \
    --zone="$ZONE" --tags=animeloom 2>/dev/null || true

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=== Setup Complete ==="
echo "SSH:   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "API:   http://$EXTERNAL_IP:8080"
echo "Stop:  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "Cost:  ~\$0.35/hour (\$300 credits = 850+ hours)"
