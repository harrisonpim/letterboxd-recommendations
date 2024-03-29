from pathlib import Path
import pulumi
import pulumi_aws as aws
from pulumi import Config

config = Config()
PROJECT = config.require("project")
PUBLIC_KEY_PATH = Path.home() / ".ssh" / "letterboxd.pub"

with open(PUBLIC_KEY_PATH, "r") as key_file:
    public_key = key_file.read()

key_pair = aws.ec2.KeyPair(f"{PROJECT}-keypair", public_key=public_key)

security_group = aws.ec2.SecurityGroup(
    resource_name=f"{PROJECT}-security-group",
    description="Enable SSH access and all outbound traffic",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=22,
            to_port=22,
            cidr_blocks=["0.0.0.0/0"],
        )
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
        )
    ],
)

instance = aws.ec2.Instance(
    resource_name=f"{PROJECT}-instance",
    instance_type="p2.xlarge",
    ami="ami-0b75df58d712725de",  # ubuntu deep learning AMI
    key_name=key_pair.key_name,
    vpc_security_group_ids=[security_group.id],
    ebs_block_devices=[
        aws.ec2.InstanceEbsBlockDeviceArgs(
            device_name="/dev/sda1",
            volume_size=64,
            volume_type="gp2",
            delete_on_termination=True,
        ),
    ],
)

pulumi.export("KEY_PAIR", key_pair.id)
pulumi.export("PUBLIC_KEY_PATH", PUBLIC_KEY_PATH.as_posix())
pulumi.export("PRIVATE_KEY_PATH", PUBLIC_KEY_PATH.with_suffix("").as_posix())
pulumi.export("INSTANCE_ID", instance.id)
pulumi.export("PUBLIC_IP", instance.public_ip)
pulumi.export("EC2_PUBLIC_DNS", instance.public_dns)
