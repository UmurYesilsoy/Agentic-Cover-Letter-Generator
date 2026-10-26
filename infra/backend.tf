terraform {
  backend "azurerm" {
    storage_account_name = "tfstateumur7444"
    container_name       = "tfstate"
    key                  = "agentic-cover-letter.tfstate"

    use_azuread_auth = true
    use_cli          = true
  }
}
