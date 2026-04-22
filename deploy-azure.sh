#!/usr/bin/env bash
# simple helper to build & deploy the container to Azure Web App
# usage: ./deploy-azure.sh <resource-group> <acr-name> <app-name>

RG=$1
ACR=$2
APP=$3

if [ -z "$RG" ] || [ -z "$ACR" ] || [ -z "$APP" ]; then
  echo "Usage: $0 <resource-group> <acr-name> <app-name>"
  exit 1
fi

# ensure azure cli logged in
az account show >/dev/null 2>&1 || az login

# build and push image to ACR
az acr build --registry "$ACR" --image "$APP:latest" .

# create App Service plan if necessary (basic tier)
az appservice plan create --name "${APP}-plan" --resource-group "$RG" --sku B1 --is-linux || true

# create or update webapp
az webapp create --resource-group "$RG" --plan "${APP}-plan" --name "$APP" --deployment-container-image-name "$ACR.azurecr.io/$APP:latest" || \
    az webapp config container set --resource-group "$RG" --name "$APP" --docker-custom-image-name "$ACR.azurecr.io/$APP:latest"

# set startup command (Gunicorn)
az webapp config set --resource-group "$RG" --name "$APP" --startup-file "gunicorn -w 4 -b 0.0.0.0:5000 app:app"

echo "Deployment complete. Visit https://$APP.azurewebsites.net"
