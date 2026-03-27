# Manual Docker commands

## 1) Create the networks

```bash
docker network create public_net
docker network create --internal private_net
```

## 2) Create the persistent database volume

```bash
docker volume create pgdata
```

## 3) Build the custom images

```bash
docker build -t joud-rag-service ./rag_service
docker build -t joud-agent-gateway ./gateway
```

## 4) Run the database on the private network only

```bash
docker run -d --name agronomy-db --network private_net -e POSTGRES_DB=agronomy -e POSTGRES_USER=agronomy -e POSTGRES_PASSWORD=agronomy -v pgdata:/var/lib/postgresql/data postgres:16-alpine
```

## 5) Run the RAG service on the private network only

```bash
docker run -d --name rag-service --network private_net joud-rag-service
```

## 6) Run the gateway on the public network and publish port 8000

```bash
docker run -d --name agent-gateway --network public_net -p 8000:8000 -e DB_HOST=agronomy-db -e DB_PORT=5432 -e DB_NAME=agronomy -e DB_USER=agronomy -e DB_PASSWORD=agronomy -e RAG_URL=http://rag-service:8001 joud-agent-gateway
```

## 7) Connect the gateway to the private network too

```bash
docker network connect private_net agent-gateway
```

## 8) Test the system

### Health check
```bash
curl http://localhost:8000/health
```

# Ask question
$body = @{message = "My zucchini leaves have white powdery spots. What should I do first?"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body $body

# Save note
$body = @{message = "Save a note that zucchini plot A likely has powdery mildew."} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body $body

# List notes
$body = @{message = "Use SQL to show all saved notes."} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body $body
```

## 9) Show networks

```bash
docker network ls
docker network inspect public_net
docker network inspect private_net
```

## 10) Stop and remove containers

```bash
docker stop agent-gateway rag-service agronomy-db
docker rm agent-gateway rag-service agronomy-db
```

## 11) Optional cleanup

```bash
docker volume rm pgdata
docker network rm public_net private_net
```
