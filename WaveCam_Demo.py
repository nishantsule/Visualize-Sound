# Importing the relevant packages
import numpy as np
import cv2

freq = 15000
nm = 2
c0 = (346.13, 0)
rho = (1.2, 1.0e6)

if np.amin(c0) == 0:
    wavelmin = 300 / 20000.0
else:
    wavelmin = np.amin(c0) / 20000.0


# The main class that defines all constants, variables, and functions
class fdtdVar:
    def __init__(self, rs, cs):
        # Constants
        cn = 0.9 / np.sqrt(2.0)  # Courant number
        # Variables
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
        # self.mbndry = np.zeros(temp)  # image array for media block
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
        fwhmc = 2
        fwhmr = fwhmc
        self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T

    def source(self, nt):
        rm = self.r
        cm = self.c
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
        self.mvx.fill(0)
        self.mvy.fill(0)
        self.mpr.fill(0)

# Create VideoCapture object
cap = cv2.VideoCapture(0)

columns = 640
rows = 480

cap.set(3, columns)
cap.set(4, rows)

fs = fdtdVar(rows, columns)


tc = 0
while True:
    # Reset domain
    fs.update_domain()

    # Capture frame
    retval, frame = cap.read()

    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    # Update image with FDTD solution
    fs.fdtd_update()
    fs.source(tc)
    fs.boundary()
    imgdisp = img + fs.pr

    tc = tc + 1

    # Display image
    cv2.imshow("frame", imgdisp)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
