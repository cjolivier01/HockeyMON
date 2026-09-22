"""Hermetic Ubuntu 22.04 glibc sysroot for portable Rust GUI linking."""

_PACKAGES = {
    "x86_64": [
        (
            "https://launchpad.net/ubuntu/+archive/primary/+files/libc6_2.35-0ubuntu3.14_amd64.deb",
            "4aa2feae34cb6296e133af5c7429756ab5606549cd16a9d26a0060e010214523",
        ),
        (
            "https://launchpad.net/ubuntu/+archive/primary/+files/libc6-dev_2.35-0ubuntu3.14_amd64.deb",
            "49583c0ebf761a2ade6d024949e8b1d70746653cce781e3a4eb68b4dd7ced540",
        ),
    ],
    "aarch64": [
        (
            "https://launchpad.net/ubuntu/+archive/primary/+files/libc6_2.35-0ubuntu3.14_arm64.deb",
            "e4eb1c12810ccbb8758b6cc8e49b7090bf6a28f11bc1bec09a9f8ded55c7cd37",
        ),
        (
            "https://launchpad.net/ubuntu/+archive/primary/+files/libc6-dev_2.35-0ubuntu3.14_arm64.deb",
            "21e50e76729c36891e4bd4397d4f58dabe8589e972535765da3deabcc5a55bba",
        ),
    ],
}


def _run(repository_ctx, arguments, description):
    result = repository_ctx.execute(arguments, quiet = True)
    if result.return_code != 0:
        fail("{} failed: {}".format(description, result.stderr.strip()))


def _ubuntu_glibc_repository_impl(repository_ctx):
    arch = repository_ctx.os.arch
    if arch in ("amd64", "x86_64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        fail("Ubuntu 22.04 glibc sysroot does not support host architecture {}".format(arch))

    for index, package in enumerate(_PACKAGES[arch]):
        deb_path = "glibc-{}.deb".format(index)
        repository_ctx.download(
            url = package[0],
            output = deb_path,
            sha256 = package[1],
        )
        _run(repository_ctx, ["ar", "x", deb_path], "extracting {}".format(deb_path))
        _run(
            repository_ctx,
            ["tar", "--zstd", "-xf", "data.tar.zst"],
            "unpacking {}".format(deb_path),
        )
        repository_ctx.delete(deb_path)
        repository_ctx.delete("control.tar.zst")
        repository_ctx.delete("data.tar.zst")
        repository_ctx.delete("debian-binary")

    repository_ctx.file(".sysroot_marker", "Ubuntu 22.04 glibc 2.35\n")
    repository_ctx.file(
        "BUILD.bazel",
        """
exports_files([".sysroot_marker"])

filegroup(
    name = "sysroot",
    srcs = glob([
        "lib/**",
        "lib64/**",
        "usr/include/**",
        "usr/lib/**",
    ]),
    visibility = ["//visibility:public"],
)
""",
    )


ubuntu_glibc_repository = repository_rule(
    implementation = _ubuntu_glibc_repository_impl,
)
