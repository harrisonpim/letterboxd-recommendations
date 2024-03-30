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

ingress_rule = aws.ec2.SecurityGroupIngressArgs(
    description="Allow SSH access",
    protocol="tcp",
    from_port=22,
    to_port=22,
    cidr_blocks=["0.0.0.0/0"],
)

egress_rule = aws.ec2.SecurityGroupEgressArgs(
    description="Allow all outbound traffic",
    protocol="-1",
    from_port=0,
    to_port=0,
    cidr_blocks=["0.0.0.0/0"],
)

security_group = aws.ec2.SecurityGroup(
    resource_name=f"{PROJECT}-security-group",
    ingress=[ingress_rule],
    egress=[egress_rule],
)

instance = aws.ec2.Instance(
    resource_name=f"{PROJECT}-instance",
    instance_type="g4dn.xlarge",
    ami="ami-0ea3af21a2bc4db3a",  # ubuntu deep learning AMI, compatible with g4dn.xlarge
    key_name=key_pair.key_name,
    vpc_security_group_ids=[security_group.id],
)

pulumi.export("KEY_PAIR", key_pair.id)
pulumi.export("PUBLIC_KEY_PATH", PUBLIC_KEY_PATH.as_posix())
pulumi.export("PRIVATE_KEY_PATH", PUBLIC_KEY_PATH.with_suffix("").as_posix())
pulumi.export("INSTANCE_ID", instance.id)
pulumi.export("PUBLIC_IP", instance.public_ip)
pulumi.export("EC2_PUBLIC_DNS", instance.public_dns)
