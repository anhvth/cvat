docker-compose down --remove-orphans
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up  --build -d