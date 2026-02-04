import os
import requests

from core.utils import sign_streetview_url

PANO_ID = "InByN14NBfMifqXhV45hVw"
HEADING = "264.8282"
SIZE = "512x512"
FOV = "90"
PITCH = "30"
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
SIGNING_SECRET = os.environ.get("STREETVIEW_SIGNING_SECRET", "")


def main() -> None:
    unsigned_url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={SIZE}&pano={PANO_ID}&heading={HEADING}&fov={FOV}&pitch={PITCH}&key={API_KEY}"
    )
    signed_url = sign_streetview_url(unsigned_url, SIGNING_SECRET)

    print("Unsigned URL:")
    print(unsigned_url)
    print()
    print("Signed URL:")
    print(signed_url)

    try:
        resp = requests.get(signed_url, timeout=10)
        print()
        print(f"HTTP status: {resp.status_code}")
        print(f"Content length: {len(resp.content)} bytes")
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()

