# Infrastructure

Uses pulumi to spin up a GPU in AWS for training models.

## Setup

Generate a new SSH key for connecting to the GPU instance.

```sh
ssh-keygen -t rsa -b 2048 -C "YOUR_EMAIL_ADDRESS" -f ~/.ssh/letterboxd
```

And add the key to the keychain

```sh
ssh-add ~/.ssh/letterboxd
```

## Commands

Then run make to deploy the infrastructure

- `make infra-start` will deploy the infrastructure
- `make infra-stop` will destroy the infrastructure
- `make infra-export` will export a set of environment variables which can be used to connect to the instance
