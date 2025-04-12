import yt_dlp
import argparse

# URL of the YouTube video
video_url = "https://www.youtube.com/watch?v=ZItUSNL5Ies"
parser = argparse.ArgumentParser(description="Download yt video without audio")
print(f"Downloading yt video at {video_url}. Specify input yt url with --url argument when running the script to run on a different video")
parser.add_argument("--url", default=video_url, type=str)
parser.add_argument("--output_path", required=True, type=str)
args = parser.parse_args()

if not args.output_path.endswith(".mp4"):
    # TODO: Accept different formats in the future
    raise ValueError("Output file must end with mp4")

# Define options for yt-dlp
ydl_opts = {
    'format': 'bv[ext=mp4]/bv',  # Downloads best video quality but no audio
    'outtmpl': args.output_path,
    'postprocessors': [
        {
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }
    ]
}

# Download video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([args.url])
