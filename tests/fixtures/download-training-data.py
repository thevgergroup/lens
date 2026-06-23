#!/usr/bin/env python3
"""
Downloads expanded training data for LR classifier calibration.

Target: ~80 non-Grok AI images + ~80 diverse real photos.

Sources:
  AI:   GitHub repos (SD, SDXL, Flux), Wikimedia Commons AI category (CC-BY-SA/CC0)
  Real: Picsum/Unsplash (CC0), Wikimedia Commons real photos (CC-BY-SA/PD)

No authentication required. All URLs are direct downloads.

Usage:
  python3 tests/fixtures/download-training-data.py
  python3 tests/fixtures/download-training-data.py --ai-only
  python3 tests/fixtures/download-training-data.py --real-only
"""

import argparse
import time
import urllib.request
import urllib.error
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
AI_DIR   = FIXTURES_DIR / "images" / "ai" / "training"
REAL_DIR = FIXTURES_DIR / "images" / "real" / "training"


def download_url(url, dest, label="", delay=0):
    if dest.exists():
        print(f"  ~ {label or dest.name} (exists)")
        return True
    if delay:
        time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LENS-downloader/1.0)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        print(f"  ✓ {label or dest.name}")
        return True
    except Exception as e:
        print(f"  ✗ {label or dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── AI images ─────────────────────────────────────────────────────────────────

SD_BASE   = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples"
SDXL_BASE = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/assets"
FLUX_BASE = "https://raw.githubusercontent.com/black-forest-labs/flux/main/assets"
WIKI      = "https://upload.wikimedia.org/wikipedia/commons"

AI_IMAGES = [
    # Stable Diffusion v1 (CompVis repo, research license)
    (f"{SD_BASE}/txt2img/000002025.png",       "sd1-txt2img-000.png"),
    (f"{SD_BASE}/txt2img/000002035.png",       "sd1-txt2img-001.png"),
    (f"{SD_BASE}/txt2img/merged-0005.png",     "sd1-txt2img-002.png"),
    (f"{SD_BASE}/img2img/mountains-1.png",     "sd1-img2img-mountains.png"),

    # SDXL / Stability-AI generative-models (Apache 2.0)
    (f"{SDXL_BASE}/000.jpg",                  "sdxl-000.jpg"),
    (f"{SDXL_BASE}/test_image.png",           "sdxl-test-image.png"),
    (f"{SDXL_BASE}/turbo_tile.png",           "sdxl-turbo-tile.png"),

    # Flux (Apache 2.0 for schnell)
    (f"{FLUX_BASE}/schnell_grid.jpg",         "flux-schnell-grid.jpg"),
    (f"{FLUX_BASE}/dev_grid.jpg",             "flux-dev-grid.jpg"),
    (f"{FLUX_BASE}/robot.webp",               "flux-robot.webp"),

    # Midjourney — Wikimedia Commons (CC-BY-SA)
    (f"{WIKI}/e/e8/%27Greenwood_Estates_Vista_City%27_by_Midjourney.jpg", "mj-greenwood-city.jpg"),
    (f"{WIKI}/c/ca/Astronaut_walking_on_Mars.png",                         "mj-astronaut-mars.png"),

    # Stable Diffusion — Wikimedia Commons (CC licenses)
    (f"{WIKI}/a/a6/Scenic_Valley_in_the_Afternoon_%28SD1.5%29.jpg",       "wiki-sd-scenic-valley.jpg"),
    (f"{WIKI}/6/63/Cyberpunk_city_created_by_Stable_Diffusion.webp",      "wiki-sd-cyberpunk-city.webp"),
    (f"{WIKI}/4/43/Android_making_a_conclusion_in_2740.png",               "wiki-sd-android.png"),
    (f"{WIKI}/0/0e/AI_golem_waiting_for_tasks_and_providing_advice.jpg",   "wiki-sd-golem.jpg"),
    (f"{WIKI}/8/87/A_robot_writing_an_apology_letter.png",                 "wiki-dalle-robot-letter.png"),
    (f"{WIKI}/4/45/A_lonely_blue_man_curled_up_in_the_fetal_position_floats_in_nothingness.png", "wiki-dalle-blue-man.png"),

    # DALL-E 3 — Wikimedia Commons
    (f"{WIKI}/4/45/Dall-e_3_%28jan_%2724%29_artificial_intelligence_image_of_a_panda_astronaut.jpg", "wiki-dalle3-panda.jpg"),
    (f"{WIKI}/9/9d/Dalle3_-_Street_Photography_NYC_2024.jpg",              "wiki-dalle3-nyc.jpg"),
    (f"{WIKI}/a/ae/DALL-E_3_generated_image_of_a_small_dog_in_the_mountains.jpg", "wiki-dalle3-dog-mountain.jpg"),
    (f"{WIKI}/b/b5/Artificial_Intelligence_in_art_%28DALL-E%29.jpg",       "wiki-dalle-art.jpg"),
    (f"{WIKI}/3/38/Astronaut_Riding_a_Horse_%28DALL-E%29.jpg",             "wiki-dalle-astronaut-horse.jpg"),

    # Flux — Wikimedia Commons
    (f"{WIKI}/f/f1/Dragon_Encounter_During_Sunset_%28FLUX_1.1_Pro_Ultra%29.webp", "wiki-flux-dragon-sunset.webp"),
    (f"{WIKI}/7/7c/The_Path_to_the_Mountain_%28FLUX.2_Pro%29.webp",        "wiki-flux-mountain-path.webp"),
    (f"{WIKI}/9/9f/FluxSchnell_image_of_the_Eiffel_Tower.jpg",             "wiki-flux-eiffel.jpg"),

    # Ideogram — Wikimedia Commons
    (f"{WIKI}/f/fe/Welcome_to_the_Simulated_Universe_%28Ideogram_3.0%29.webp", "wiki-ideogram-universe.webp"),
    (f"{WIKI}/4/44/Scenic_Valley_in_the_Afternoon_%28Ideogram_2.0%29.webp",    "wiki-ideogram-valley.webp"),

    # Midjourney — Wikimedia Commons (more)
    (f"{WIKI}/7/73/Midjourney_Sunflower_Field.jpg",                        "wiki-mj-sunflower.jpg"),
    (f"{WIKI}/5/52/Midjourney_AI_art.jpg",                                  "wiki-mj-art.jpg"),
    (f"{WIKI}/7/7c/Midjourney_Taj_Mahal.jpg",                               "wiki-mj-taj-mahal.jpg"),
    (f"{WIKI}/1/17/A_hyperrealistic_painting_of_a_Tyrannosaurus_rex_in_a_museum_%28Midjourney%29.jpg", "wiki-mj-trex.jpg"),
    (f"{WIKI}/c/c3/Civai_midjourney_nature.jpg",                            "wiki-mj-nature.jpg"),
    (f"{WIKI}/d/d4/Midjourney_generated_image_of_a_city.jpg",               "wiki-mj-city.jpg"),
    (f"{WIKI}/6/61/Midjourney_fantasy_castle.jpg",                          "wiki-mj-castle.jpg"),

    # Stable Diffusion — Wikimedia Commons (more)
    (f"{WIKI}/e/e4/Stable_diffusion%2C_an_artificial_female_generated_with_artificial_intelligence.jpg", "wiki-sd-female.jpg"),
    (f"{WIKI}/2/2b/AI_generated_image_of_a_fantasy_landscape_Stable_Diffusion.jpg", "wiki-sd-fantasy.jpg"),
    (f"{WIKI}/2/2f/Generative_AI_art_of_a_woman%27s_face.jpg",              "wiki-sd-woman-face.jpg"),
    (f"{WIKI}/5/5e/2023_AI_image_of_a_husky.jpg",                           "wiki-sd-husky.jpg"),
]


# ── Real images ───────────────────────────────────────────────────────────────

PICSUM_IDS = [
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    30, 31, 32, 33, 34, 37, 39, 40, 42, 43,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 66, 67, 68, 69, 70,
]

# Wikimedia Commons real photos — full-resolution URLs (no /thumb/ prefix)
REAL_IMAGES = [
    (f"{WIKI}/a/a7/Camponotus_flavomarginatus_ant.jpg",             "wiki-ant.jpg"),
    (f"{WIKI}/3/3a/Cat03.jpg",                                       "wiki-cat.jpg"),
    (f"{WIKI}/2/26/YellowLabradorLooking_new.jpg",                   "wiki-labrador.jpg"),
    (f"{WIKI}/4/41/Sunflower_from_Silesia2.jpg",                     "wiki-sunflower.jpg"),
    (f"{WIKI}/1/1a/24701-nature-natural-beauty.jpg",                 "wiki-ocean.jpg"),
    (f"{WIKI}/1/1e/Stonehenge.jpg",                                  "wiki-stonehenge.jpg"),
    (f"{WIKI}/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg",               "wiki-eiffel.jpg"),
    (f"{WIKI}/f/f6/Tigerramki.jpg",                                  "wiki-tiger.jpg"),
    (f"{WIKI}/1/14/Gatto_europeo4.jpg",                              "wiki-cat2.jpg"),
    (f"{WIKI}/4/40/Sunflower_sky_backdrop.jpg",                      "wiki-sunflower2.jpg"),
    (f"{WIKI}/6/6b/American_Beaver.jpg",                             "wiki-beaver.jpg"),
    (f"{WIKI}/0/05/Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg", "wiki-central-park.jpg"),
    (f"{WIKI}/3/36/Hopetoun_falls.jpg",                              "wiki-waterfall.jpg"),
    (f"{WIKI}/7/73/Lion_waiting_in_Namibia.jpg",                     "wiki-lion.jpg"),
    (f"{WIKI}/a/a4/Laughing_kookaburra_dec08_06.jpg",                "wiki-kookaburra.jpg"),
    (f"{WIKI}/2/29/Matterhorn_from_Domh%C3%BCtte_-_2.jpg",           "wiki-matterhorn.jpg"),
    (f"{WIKI}/b/b5/Polar_bear_%28Ursus_maritimus%29_with_cub.jpg",   "wiki-polar-bear.jpg"),
    (f"{WIKI}/0/0c/GoldenGateBridge-001.jpg",                        "wiki-golden-gate.jpg"),
    (f"{WIKI}/4/4d/Cat_November_2010-1a.jpg",                        "wiki-cat3.jpg"),
    (f"{WIKI}/9/9b/Gustav_chocolate.jpg",                            "wiki-chocolate.jpg"),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-only",   action="store_true")
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()

    do_ai   = not args.real_only
    do_real = not args.ai_only

    if do_ai:
        AI_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n=== AI images → {AI_DIR} ===")
        ok = 0
        for url, name in AI_IMAGES:
            delay = 3 if "wikimedia.org" in url else 0
            ok += download_url(url, AI_DIR / name, name, delay=delay)
        total = len(list(AI_DIR.glob("*.[jpwJPW]*")))
        print(f"\n  {ok}/{len(AI_IMAGES)} downloaded this run, {total} total in dir")

    if do_real:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Real images → {REAL_DIR} ===")

        print("\n  Picsum/Unsplash (CC0):")
        ok_p = 0
        for pid in PICSUM_IDS:
            name = f"picsum-{pid:03d}.jpg"
            if download_url(f"https://picsum.photos/id/{pid}/800/600.jpg", REAL_DIR / name, name):
                ok_p += 1

        print(f"\n  Wikimedia Commons real photographs (3s delay between requests):")
        ok_w = sum(download_url(url, REAL_DIR / name, name, delay=3) for url, name in REAL_IMAGES)

        total = len(list(REAL_DIR.glob("*.[jpwJPW]*")))
        print(f"\n  Picsum: {ok_p}/{len(PICSUM_IDS)}, Wikimedia: {ok_w}/{len(REAL_IMAGES)}, {total} total in dir")

    print("\nDone.")


if __name__ == "__main__":
    main()
