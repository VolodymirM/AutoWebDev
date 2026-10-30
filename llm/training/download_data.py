"""Download the Multi-SWE-bench dataset from Hugging Face."""

from importlib import import_module
from pathlib import Path
import subprocess
from shutil import rmtree, which

REPOSITORY_URLS = (
	"https://github.com/pamarthidurgarao/java-angular-web-app",
	"https://github.com/cavanosa/tutorial-keycloak-angular-spring-boot",
	"https://github.com/sanchit2107/Flight-Management-System",
	"https://github.com/JavaInUse/angular7-crud",
	"https://github.com/dinesh-more/e-commerce-fullstack_springboot_angular_aws",
	"https://github.com/BraianS/fullstack-spring-boot-and-angular-blog",
	"https://github.com/adnanrahin/Movie-Application",
	"https://github.com/monster0318/fullstack-oauth2-angular-spring-boot-keycloak",
	"https://github.com/joedayz/BuildFullStackSpringBootAndAngular",
	"https://github.com/xalio-dev/speak-your-mind",
	"https://github.com/saber-barhoumi/SpeedyGo-SpringBoot-Angular",
	"https://github.com/EmanuelSergio/SpringBoot-Angular-FullStack",
	"https://github.com/happylifedev/Microservice-eShop-App",
	"https://github.com/RafrafiNawress/Full-stack-Angular-SpringBoot",
	"https://github.com/VishalPawar1010/SpringBoot_Angular_FullStackWebApp",
	"https://github.com/OliviaBennQ/SpringBoot-Angular-Items-CRUD",
	"https://github.com/MPARSURAM99/spring-angular-E-com-app",
	"https://github.com/TracVietHustCS/Spring-full-stack",
	"https://github.com/mrin9/Angular-SpringBoot-REST-JWT",
	"https://github.com/iamtech/SpringBoot-Microservice-Eurka-Zuul",
	"https://github.com/mbachmann/angular-20-jwt-auth-starter-standalone",
	"https://github.com/AishwaryaGhosh67/JWTTokenAuthentication",
	"https://github.com/jiyoonpark05/spring-boot-jwt-authentication",
	"https://github.com/thekingsimba/fullstack-Java-Spring-boot-and-angular-app-code-sample",
	"https://github.com/elijahbowman/spring-boot-angular-ecommerce-microservices-example",
	"https://github.com/deepakbhalla/microservices-springboot-1",
	"https://github.com/JFWenisch/elasticsearch-microservice-example",
)


def download_repositories(destination: Path) -> None:
	if which("git") is None:
		raise SystemExit("Install Git and make sure it is available on PATH.")

	destination.mkdir(parents=True, exist_ok=True)
	existing_names = set()
	for directory in destination.rglob("*"):
		if directory.is_dir():
			name = directory.name.casefold()
			for suffix in ("-master", "-main", "-develop"):
				if name.endswith(suffix):
					name = name[: -len(suffix)]
					break
			existing_names.add(name)

	failed_repositories = []
	for repository_url in REPOSITORY_URLS:
		repository_name = repository_url.rstrip("/").rsplit("/", 1)[-1]
		target = destination / repository_name
		if repository_name.casefold() in existing_names:
			print(f"Repository already exists, skipping: {repository_name}")
			continue

		try:
			subprocess.run(
				["git", "clone", "--depth", "1", repository_url, str(target)],
				check=True,
			)
		except (OSError, subprocess.CalledProcessError) as exc:
			print(f"Failed to clone {repository_url}: {exc}")
			failed_repositories.append(repository_name)
		else:
			existing_names.add(repository_name.casefold())

	if failed_repositories:
		print("Repositories not downloaded: " + ", ".join(failed_repositories))

def main() -> None:
	try:
		snapshot_download = import_module("huggingface_hub").snapshot_download
	except ImportError as exc:
		raise SystemExit("Install the dependency with: python -m pip install huggingface_hub") from exc

	destination = Path(__file__).resolve().parent / "automationexamples"
	snapshot_download(
		repo_id="ByteDance-Seed/Multi-SWE-bench",
		repo_type="dataset",
		local_dir=str(destination),
		allow_patterns=["java/**", "js/**", "ts/**"],
	)

	metadata_dir = destination / ".cache" / "huggingface"
	if metadata_dir.exists():
		rmtree(metadata_dir)
		if not any(metadata_dir.parent.iterdir()):
			metadata_dir.parent.rmdir()

	print(f"Dataset downloaded to: {destination}")
	download_repositories(Path(__file__).resolve().parent / "trainingrepos")


if __name__ == "__main__":
	main()