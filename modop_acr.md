Enable admin user on the ACR using: az acr update -n isschat --admin-enabled true
Get ACR credentials using: az acr credential show --name isschat
Login Docker to the ACR using: docker login isschat.azurecr.io

