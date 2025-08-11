#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

IMAGE_NAME="priyamtiwarigenai/mlflow-server:latest"
CONTAINER_NAME="mlflow-server"
PORT=5000

echo "🚀 Starting deployment..."

# Detect environment
if curl -s http://169.254.169.254/latest/meta-data/ >/dev/null 2>&1; then
    echo "📡 Detected EC2 instance."
    ENVIRONMENT="EC2"
else
    echo "💻 Detected Local machine."
    ENVIRONMENT="LOCAL"
fi

# Pull the latest Docker image
echo "📥 Pulling latest Docker image: $IMAGE_NAME"
docker pull $IMAGE_NAME

# Stop and remove any existing container
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🛑 Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME || true
    echo "🗑 Removing old container..."
    docker rm $CONTAINER_NAME || true
fi

# Run the container
echo "▶️ Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p ${PORT}:${PORT} \
    $IMAGE_NAME \
    mlflow ui --host 0.0.0.0 --port ${PORT}

echo "✅ Deployment complete."

if [ "$ENVIRONMENT" == "LOCAL" ]; then
    echo "🌐 Access the app at: http://localhost:${PORT}"
else
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
    echo "🌐 Access the app at: http://${PUBLIC_IP}:${PORT}"
fi
