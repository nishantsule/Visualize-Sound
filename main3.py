# Importing the relevant packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import pyaudio

#### Ask for user input
plt.ion()
dflag = input("Enter 'd' to run the default setup or 'c' for customizing: ")
print("")
if dflag == "d":
    freq = 15000
    nm = 2
    c0 = (346.13, 0)
    rho = (1.2, 1.0e6)
    stype = "point"
    mflag = "n"
    sflag = "n"
    print("Point source of 15000 Hz in a closed domain with air at 295 K using the default camera resolution")
    print("")
elif dflag == "c":
    sflag = input("Enter 'a' to record an audio source or 'n' to enter a numeric source frequency: ")
    print("")
    if sflag == "a":
        # Initialize portaudio
        p = pyaudio.PyAudio()
        numdev = p.get_device_count() - 1
        if (numdev < 1):
            sys.exit("Error: You do not have the hardware to record audio")
        else:
            freq = 0
            rt = 80000
            rec_sec = 1
            chsize = 1024
            input("Start the audio and then press Enter to record...")
            print("")
            astream = p.open(format=pyaudio.paInt16, channels=1, rate=rt, input=True, frames_per_buffer=chsize)
            # Record audio
            frames = []
            for i in range(0, np.int(rt / chsize * rec_sec)):
                adata = astream.read(chsize)
                frames.append(np.fromstring(adata, dtype=np.int16))
            # Close audio stream
            astream.stop_stream()
            astream.close()
            p.terminate()
            atemp = np.asfarray(np.hstack(frames))
            skipf = int(np.alen(atemp) / 4)
            last = int(np.alen(atemp))
            asource = ((atemp[skipf:last - skipf] - np.mean(atemp[skipf:last - skipf]))
                       / (np.amax(atemp[skipf:last - skipf]) - np.amin(atemp[skipf:last - skipf])))
            sfreqspec = np.fft.rfft(asource)
            time = np.linspace(0, last, last) * rec_sec / rt
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("rescaled amplitude (arb units)")
            ax1.set_title("Recorded audio")
            ax1.plot(time[skipf:last - skipf], asource)
            fig1.tight_layout()
            fig1.show()
            fig1.canvas.draw()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.set_xlabel('freq (Hz)')
            ax2.set_ylabel('power (arb units)')
            ax2.set_title("Power spectrum of recorded audio")
            ax2.plot(np.abs(sfreqspec)**2)
            fig2.tight_layout()
            fig2.show()
            fig2.canvas.draw()
    elif sflag == "n":
        try:
            freq = float(input("Enter the sound frequency in Hz (20-20000): "))
            print("")
        except ValueError:
            sys.exit("Error: enter a number between 20 and 20000")
    else:
        sys.exit("Error: press 'a' to record an audio source or n to enter a numeric source frequency")
    vs = input("Enter sound velocity in m/s (If medium is air or water type 'air' or 'water'): ")
    print("")
    if vs == "air":
        c0 = (346.13, 0)
        rho = (1.2, 1.0e6)
    elif vs == "water":
        c0 = (1481, 0)
        rho = (1000, 1.0e6)
    else:
        try:
            c0 = (float(vs), 0)
            mdensity = float(input("Enter density of medium in kg/m^3: "))
            print("")
            rho = (mdensity, 1.0e6)
        except ValueError:
            sys.exit("Error: enter a numeric value, or 'air', or 'water' ")
    stype = input("Enter 'point' or 'line' for type of source: ")
    print("")
    if stype != "line" and stype != "point":
        sys.exit("Error: type either point or line for type of source")
    mflag = input("Do you want to insert a block of medium at a different temperature? (y/n): ")
    print("")
    if mflag == "y":
        nm = 3
        temparature = float(input("Enter the absolute temperature of the block in K (50-500): "))
        print("")
        ct = c0[0] * np.sqrt(temparature/293)
        c0 = (c0[0], 0, ct)
        rho = (rho[0], 1.0e6, rho[0])
    elif mflag == "n":
        nm = 2
    else:
        sys.exit("Error: enter 'y' to insert a block of different temperature or enter 'n' ")
else:
    sys.exit("Error: enter 'd' for default setup or 'c' to customize")

if np.amin(c0) == 0:
    wavelmin = 300 / 20000.0
else:
    wavelmin = np.amin(c0) / 20000.0


#### The main class that defines all constants, variables, and functions
class fdtdVar:
    def __init__(self, rs, cs):
        ## Constants
        cn = 0.9 / np.sqrt(2.0)  # Courant number
        ## Variables
        self.r = np.int(rs)  # number of rows
        self.c = np.int(cs)  # number of columns
        self.freq = freq  # frequency of source
        temp = (self.r, self.c + 1)
        self.vx = np.zeros(temp)  # velocity along x
        self.mvx = np.zeros(temp, dtype=np.int8)
        temp = (self.r + 1, self.c)
        self.vy = np.zeros(temp)  # velocity along y
        self.mvy = np.zeros(temp, dtype=np.int8)
        temp = (self.r, self.c)
        self.pr = np.zeros(temp)  # pressure
        self.mbndry = np.zeros(temp)  # image array for media block
        self.gaussamp = np.zeros(temp)
        self.mpr = np.zeros(temp, dtype=np.int8)
        self.dx = wavelmin/10.0  # grid cell size
        self.dt = cn * self.dx / np.amax(c0)  # time step size
        self.ca = np.ones(nm)
        self.cb = np.ones(nm)
        self.da = np.ones(nm)
        self.db = np.ones(nm)
        for i in range(0, nm, 1):
            self.cb[i] = c0[i] ** 2 * rho[i] * self.dt / self.dx
            self.db[i] = self.dt / (rho[i] * self.dx)
        self.da[1] = 0
        temp = (self.r, 2, 2)
        self.vxl = np.zeros(temp)
        self.vxr = np.zeros(temp)
        temp = (self.c, 2, 2)
        self.vyb = np.zeros(temp)
        self.vyt = np.zeros(temp)
        print("dx [m] = ", self.dx)
        print("dt [s] = ", self.dt)
        print("")
        rtemp = np.arange(0, self.r, 1)
        ctemp = np.arange(0, self.c, 1)
        rm, cm = np.meshgrid(rtemp, ctemp)
        rc = np.int(self.r / 2)
        cc = np.int(self.c / 2)
        if stype == "point":
            fwhmc = 2
            fwhmr = fwhmc
            self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T
        elif stype == "line":
            fwhmc = 2
            fwhmr = 16
            self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T
    
    def source(self, nt):
        rm = self.r
        cm = self.c
        # prs = self.dx * np.sin(2 * pi * self.freq * nt * self.dt) / self.cb[0]
        if sflag == "a":
            if nt < np.alen(asource):
                prs = self.dx * asource[nt] / self.cb[0]
            else:
                prs = 0
        else:
            prs = self.dx * np.sin(2 * np.pi * self.freq * nt * self.dt) / self.cb[0]

        # Update pressure with source
        self.pr[1:rm - 1, 1:cm - 1] = (self.pr[1:rm - 1, 1:cm - 1]
                                       - self.cb[self.mpr[1:rm - 1, 1:cm - 1]] * prs
                                       * self.gaussamp[1:rm - 1, 1:cm - 1] / self.dx)

    def fdtd_update(self):
        ri = self.r
        ci = self.c
        self.pr[0:ri, 0:ci] = (self.ca[self.mpr[0:ri, 0:ci]] * self.pr[0:ri, 0:ci]
                               - self.cb[self.mpr[0:ri, 0:ci]]
                               * ((self.vx[0:ri, 1:ci + 1] - self.vx[0:ri, 0:ci])
                                  + (self.vy[1:ri + 1, 0:ci] - self.vy[0:ri, 0:ci])))
        self.vx[0:ri, 1:ci] = (self.da[self.mvx[0:ri, 1:ci]] * self.vx[0:ri, 1:ci]
                               - self.db[self.mvx[0:ri, 1:ci]] * (self.pr[0:ri, 1:ci] - self.pr[0:ri, 0:ci - 1]))
        self.vy[1:ri, 0:ci] = (self.da[self.mvy[1:ri, 0:ci]] * self.vy[1:ri, 0:ci]
                               - self.db[self.mvy[1:ri, 0:ci]] * (self.pr[1:ri, 0:ci] - self.pr[0:ri - 1, 0:ci]))

    def boundary(self):
        ri = self.r
        ci = self.c
        c1 = (c0[0] * self.dt - self.dx) / (c0[0] * self.dt + self.dx)
        c2 = 2 * self.dx / (c0[0] * self.dt + self.dx)
        c3 = (c0[0] * self.dt) ** 2 / (2 * self.dx * (c0[0] * self.dt + self.dx))
        # Left and right boundaries
        self.vx[1:ri - 1, 0] = (-self.vxl[1:ri - 1, 1, 1]
                                + c1 * (self.vx[1:ri - 1, 1] + self.vxl[1:ri - 1, 0, 1])
                                + c2 * (self.vxl[1:ri - 1, 0, 0] + self.vxl[1:ri - 1, 1, 0])
                                + c3 * (self.vxl[2:ri, 0, 0] - 2 * self.vxl[1:ri - 1, 0, 0]
                                        + self.vxl[0:ri - 2, 0, 0] + self.vxl[2:ri, 1, 0]
                                        - 2 * self.vxl[1:ri - 1, 1, 0] + self.vxl[0:ri - 2, 1, 0]))
        self.vx[1:ri - 1, ci] = (-self.vxr[1:ri - 1, 1, 1]
                                 + c1 * (self.vx[1:ri - 1, ci - 1] + self.vxr[1:ri - 1, 0, 1])
                                 + c2 * (self.vxr[1:ri - 1, 0, 0] + self.vxr[1:ri - 1, 1, 0])
                                 + c3 * (self.vxr[2:ri, 0, 0] - 2 * self.vxr[1:ri - 1, 0, 0]
                                         + self.vxr[0:ri - 2, 0, 0] + self.vxr[2:ri, 1, 0]
                                         - 2 * self.vxr[1:ri - 1, 1, 0] + self.vxr[0:ri - 2, 1, 0]))

        # Bottom and top boundaries
        self.vy[0, 1:ci - 1] = (-self.vyb[1:ci - 1, 1, 1]
                                + c1 * (self.vy[1, 1:ci - 1] + self.vyb[1:ci - 1, 0, 1])
                                + c2 * (self.vyb[1:ci - 1, 0, 0] + self.vyb[1:ci - 1, 1, 0])
                                + c3 * (self.vyb[2:ci, 0, 0] - 2 * self.vyb[1:ci - 1, 0, 0]
                                        + self.vyb[0:ci - 2, 0, 0] + self.vyb[2:ci, 1, 0]
                                        - 2 * self.vyb[1:ci - 1, 1, 0] + self.vyb[0:ci - 2, 1, 0]))
        self.vy[ri, 1:ci - 1] = (-self.vyt[1:ci - 1, 1, 1]
                                 + c1 * (self.vy[ri - 1, 1:ci - 1] + self.vyt[1:ci - 1, 0, 1])
                                 + c2 * (self.vyt[1:ci - 1, 0, 0] + self.vyt[1:ci - 1, 1, 0])
                                 + c3 * (self.vyt[2:ci, 0, 0] - 2 * self.vyt[1:ci - 1, 0, 0]
                                         + self.vyt[0:ci - 2, 0, 0] + self.vyt[2:ci, 1, 0]
                                         - 2 * self.vyt[1:ci - 1, 1, 0] + self.vyt[0:ci - 2, 1, 0]))
        # Corners
        self.vx[0, 0] = self.vxl[1, 1, 1]
        self.vx[ri - 1, 0] = self.vxl[ri - 2, 1, 1]
        self.vx[0, ci] = self.vxr[1, 1, 1]
        self.vx[ri - 1, ci] = self.vxr[ri - 2, 1, 1]
        self.vy[0, 0] = self.vyb[1, 1, 1]
        self.vy[0, ci - 1] = self.vyb[ci - 2, 1, 1]
        self.vy[ri, 0] = self.vyt[1, 1, 1]
        self.vy[ri, ci - 1] = self.vyt[ci - 2, 1, 1]

        # Store boundary values
        for i in range(0, 2, 1):
            self.vxl[0:ri, i, 1] = self.vxl[0:ri, i, 0]
            self.vxl[0:ri, i, 0] = self.vx[0:ri, i]
            self.vxr[0:ri, i, 1] = self.vxr[0:ri, i, 0]
            self.vxr[0:ri, i, 0] = self.vx[0:ri, ci - i]
            self.vyb[0:ci, i, 1] = self.vyb[0:ci, i, 0]
            self.vyb[0:ci, i, 0] = self.vy[i, 0:ci]
            self.vyt[0:ci, i, 1] = self.vyt[0:ci, i, 0]
            self.vyt[0:ci, i, 0] = self.vy[ri - i, 0:ci]

    def update_domain(self):
        if mflag == "y":
            cm = self.c
            rm = self.r
            c1 = np.int(cm/2) + np.int(cm/8)
            c2 = c1 + np.int(cm/8)
            self.mvx.fill(0)
            self.mvy.fill(0)
            self.mpr.fill(0)
            self.mvx[40:rm - 40, c1:c2] = 2
            self.mvy[40:rm - 40, c1:c2] = 2
            self.mpr[40:rm - 40, c1:c2] = 2
            self.mbndry[40, c1:c2] = 0.50
            self.mbndry[rm - 40, c1:c2] = 0.50
            self.mbndry[40:rm - 40, c1] = 0.50
            self.mbndry[40:rm - 40, c2] = 0.50
        elif mflag == "n":
            self.mvx.fill(0)
            self.mvy.fill(0)
            self.mpr.fill(0)


#### Ask user for option to read an image from file or start webcam to capture a frame

vflag = input("Do you want to read an image (enter i) or capture a frame from webcam (enter w)? ")
print("")
if vflag == "w":
    ## Create VideoCapture object
    cap = cv2.VideoCapture(0)

    ## Array sizes (resolution dependent)
    retval = cap.get(3)
    columns = retval
    retval = cap.get(4)
    rows = retval

    ## Set frame resolution
    if dflag == "c" and columns > 1000:
        print("Default frame resolution ", np.int(columns), "x", np.int(rows))
        if input("Do you want to reduce resolution to increase speed? (y/n): ") == "y":
            print("")
            print("Setting new resolution to 640 x 480")
            print("")
            columns = 640
            rows = 480

    cap.set(3, columns)
    cap.set(4, rows)
elif vflag == "i":
    ## Read image
    imgname = input("Enter the filename of your image including extension: ")
    print("")
    img = cv2.imread(imgname, 0)
    rows, columns = img.shape
    ar = 640 / columns
    dim = (640, int(rows * ar))
    # Image cleanup
    img = cv2.medianBlur(img, 21)
    resimg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rows, columns = resimg.shape
    print("Resized the image to 320 pixels wide")
    print("")
    threshimg = cv2.adaptiveThreshold(resimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
else:
    sys.exit("Error: type either 'i' to read image or 'w' to read from webcam")

#### Create FDTD object
fs = fdtdVar(rows, columns)
tc = 0

if dflag == "c":
    bflag = input("Enter 'c' for a closed domain or 'o' for an open domain: ")
    print("")
    if bflag != "c" and bflag != "o":
        sys.exit("Error: enter 'c' for closed domain or 'o' for open domain")
else:
    bflag = "c"

#### Wait to start wave propagation
if vflag == "w":
    print("Click on the camera window and press 's' to start wave propagation...")
    print("")
    while True:
        # Reset domain
        fs.update_domain()

        # Capture frame
        retval, frame = cap.read()

        # Convert to grayscale
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display image`
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break 
    
    # Webcam frame cleanup   
    blurimg = cv2.medianBlur(img, 21)
    threshimg = cv2.adaptiveThreshold(blurimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
    img = threshimg / 256.0
    
    # Clean up edges
    img[0:5, 0:fs.c] = 1
    img[fs.r - 5:fs.r, 0:fs.c] = 1
    img[0:fs.r, 0:5] = 1
    img[0:fs.r, fs.c - 5:fs.c] = 1
    
    # Create rigid material
    imgtemp = np.pad(img, ((0, 0), (0, 1)), "constant", constant_values=1.0)
    idx = imgtemp < 0.4
    fs.mvx[idx] = 1
    imgtemp = np.pad(img, ((0, 1), (0, 0)), "constant", constant_values=1.0)
    idx = imgtemp < 0.4
    fs.mvy[idx] = 1   
    
    while True:
        # Update image with FDTD solution
        fs.fdtd_update()
        fs.source(tc)
        if bflag == "o":
            fs.boundary()
        imgdisp = img + fs.pr + fs.mbndry
        # Display image
        cv2.imshow("frame", imgdisp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        tc = tc + 1
    ## Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
            
elif vflag == "i":
    # Reset domain
    fs.update_domain()

    # Normalize image
    img = threshimg / 256.0
    
    # Clean up edges
    img[0:5, 0:fs.c] = 1
    img[fs.r - 5:fs.r, 0:fs.c] = 1
    img[0:fs.r, 0:5] = 1
    img[0:fs.r, fs.c - 5:fs.c] = 1
    
    # Create rigid material
    imgtemp = np.pad(img, ((0, 0), (0, 1)), "constant", constant_values=1.0)
    idx = imgtemp < 0.5
    fs.mvx[idx] = 1
    imgtemp = np.pad(img, ((0, 1), (0, 0)), "constant", constant_values=1.0)
    idx = imgtemp < 0.5
    fs.mvy[idx] = 1

    try:
        while True:
            # Update image with FDTD solution
            fs.fdtd_update()
            fs.source(tc)
            if bflag == "o":
                fs.boundary()
            imgdisp = img + fs.pr + fs.mbndry
            
            # Display image
            cv2.imshow("frame", imgdisp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            tc = tc + 1
            
    except KeyboardInterrupt:
        pass
        
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.imshow(imgdisp, cmap="gray")
fig3.tight_layout()
fig3.show()
fig3.canvas.draw()
