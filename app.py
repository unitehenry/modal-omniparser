import modal
import image
import volume

app = modal.App("omniparser")

if __name__ == "__main__":
    import parse

    parse = app.function(
        gpu="h100", image=image.omniparser_v2_0_1, volumes={"/data": volume.vol}
    )(volume.cache(parse.parse))

    with app.run():
        print(parse.remote())
