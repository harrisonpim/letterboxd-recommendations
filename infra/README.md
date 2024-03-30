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
- `make infra-export` will export a set of environment variables from pulumi which can be used to connect to the instance

## Connecting to the instance

```sh
echo ubuntu@$EC2_PUBLIC_DNS
```

Then connect to the instance in vscode by opening the remote-ssh extension and connecting to the instance using your `ubuntu@$EC2_PUBLIC_DNS` value.

## Copying files to the instance

```sh
tar czf - ./ | pv | ssh ubuntu@$EC2_PUBLIC_DNS 'mkdir -p ~/letterboxd-recommendations && tar xzf - -C ~/letterboxd-recommendations/'
```

## Copying trained model files from the instance

```sh
scp -r ec2-user@$EC2_PUBLIC_DNS:~/letterboxd-recommendations/data/models/ ./data/models/
```
