# Importing the relevant packages
import numpy as np
import cv2

freq = 20000
nm = 2
c0 = (346.13, 0)
rho = (1.2, 1.0e6)
wavelmin = c0[0] / freq


# The main class that defines all constants, variables, and functions
class fdtdVar:
    def __init__(self, rs, cs):
        # Constants
        cn = 0.9 / np.sqrt(2.0)  # Courant number
        # Variables
        self.r = np.int(rs)  # number of rows
        self.c = np.int(cs)  # number of columns
        self.freq = freq  # frequency of source
        temp = (self.r - 1, self.c)
        self.vx = np.zeros(temp)  # velocity along x
        self.mvx = np.zeros(temp, dtype=np.int8)
        temp = (self.r, self.c - 1)
        self.vy = np.zeros(temp)  # velocity along y
        self.mvy = np.zeros(temp, dtype=np.int8)
        temp = (self.r, self.c)
        self.pr = np.zeros(temp)  # pressure
        self.gaussamp = np.zeros(temp)
        self.mpr = np.zeros(temp, dtype=np.int8)
        self.dx = wavelmin/25.0  # grid cell size
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
        self.prl = np.zeros(temp)
        self.prr = np.zeros(temp)
        temp = (self.c, 2, 2)
        self.prb = np.zeros(temp)
        self.prt = np.zeros(temp)
        rtemp = np.arange(0, self.r, 1)
        ctemp = np.arange(0, self.c, 1)
        rm, cm = np.meshgrid(rtemp, ctemp)
        rc = np.int(self.r / 2)
        cc = np.int(self.c / 2)
        fwhmc = 2
        fwhmr = fwhmc
        self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T

    def source(self, nt):
        ri = self.r
        ci = self.c
        prs = self.dx * np.sin(2 * np.pi * self.freq * nt * self.dt) / self.cb[0]
        # Update pressure with source
        self.pr[1:ri - 1, 1:ci - 1] = (self.pr[1:ri - 1, 1:ci - 1]
                                       - self.cb[self.mpr[1:ri - 1, 1:ci - 1]] * prs
                                       * self.gaussamp[1:ri - 1, 1:ci - 1] / self.dx)

    def fdtd_update_pr(self):
        ri = self.r
        ci = self.c
        self.pr[1:ri - 1, 1:ci - 1] = (self.ca[self.mpr[1:ri - 1, 1:ci - 1]] * self.pr[1:ri - 1, 1:ci - 1]
                                       - self.cb[self.mpr[1:ri - 1, 1:ci - 1]]
                                       * ((self.vx[1:ri - 1, 1:ci - 1] - self.vx[0:ri - 2, 1:ci - 1])
                                       + (self.vy[1:ri - 1, 1:ci - 1] - self.vy[1:ri - 1, 0:ci - 2])))

    def fdtd_update_v(self):
        ri = self.r
        ci = self.c
        self.vx[0:ri - 1, 0:ci] = (self.da[self.mvx[0:ri - 1, 0:ci]] * self.vx[0:ri - 1, 0:ci]
                                   - self.db[self.mvx[0:ri - 1, 0:ci]]
                                   * (self.pr[1:ri, 0:ci] - self.pr[0:ri - 1, 0:ci]))
        self.vy[0:ri, 0:ci - 1] = (self.da[self.mvy[0:ri, 0:ci - 1]] * self.vy[0:ri, 0:ci - 1]
                                   - self.db[self.mvy[0:ri, 0:ci - 1]]
                                   * (self.pr[0:ri, 1:ci] - self.pr[0:ri, 0:ci - 1]))

    def boundary(self):
        ri = self.r
        ci = self.c
        c1 = (c0[0] * self.dt - self.dx) / (c0[0] * self.dt + self.dx)
        c2 = 2 * self.dx / (c0[0] * self.dt + self.dx)
        c3 = (c0[0] * self.dt) ** 2 / (2 * self.dx * (c0[0] * self.dt + self.dx))
        # Left and right boundaries
        self.pr[1:ri - 1, 0] = (-self.prl[1:ri - 1, 1, 1]
                                + c1 * (self.pr[1:ri - 1, 1] + self.prl[1:ri - 1, 0, 1])
                                + c2 * (self.prl[1:ri - 1, 0, 0] + self.prl[1:ri - 1, 1, 0])
                                + c3 * (self.prl[2:ri, 0, 0] - 2 * self.prl[1:ri - 1, 0, 0] + self.prl[0:ri - 2, 0, 0]
                                        + self.prl[2:ri, 1, 0] - 2 * self.prl[1:ri - 1, 1, 0]
                                        + self.prl[0:ri - 2, 1, 0]))
        self.pr[1:ri - 1, ci - 1] = (-self.prr[1:ri - 1, 1, 1]
                                     + c1 * (self.pr[1:ri - 1, ci - 2] + self.prr[1:ri - 1, 0, 1])
                                     + c2 * (self.prr[1:ri - 1, 0, 0] + self.prr[1:ri - 1, 1, 0])
                                     + c3 * (self.prr[2:ri, 0, 0] - 2 * self.prr[1:ri - 1, 0, 0]
                                             + self.prr[0:ri - 2, 0, 0] + self.prr[2:ri, 1, 0]
                                             - 2 * self.prr[1:ri - 1, 1, 0] + self.prr[0:ri - 2, 1, 0]))
        # Top and bottom boundaries
        self.pr[0, 1:ci - 1] = (-self.prt[1:ci - 1, 1, 1]
                                + c1 * (self.pr[1, 1:ci - 1] + self.prt[1:ci - 1, 0, 1])
                                + c2 * (self.prt[1:ci - 1, 0, 0] + self.prt[1:ci - 1, 1, 0])
                                + c3 * (self.prt[2:ci, 0, 0] - 2 * self.prt[1:ci - 1, 0, 0] + self.prt[0:ci - 2, 0, 0]
                                        + self.prt[2:ci, 1, 0] - 2 * self.prt[1:ci - 1, 1, 0]
                                        + self.prt[0:ci - 2, 1, 0]))
        self.pr[ri - 1, 1:ci - 1] = (-self.prb[1:ci - 1, 1, 1]
                                     + c1 * (self.pr[ri - 2, 1:ci - 1] + self.prb[1:ci - 1, 0, 1])
                                     + c2 * (self.prb[1:ci - 1, 0, 0] + self.prb[1:ci - 1, 1, 0])
                                     + c3 * (self.prb[2:ci, 0, 0] - 2 * self.prb[1:ci - 1, 0, 0]
                                             + self.prb[0:ci - 2, 0, 0] + self.prb[2:ci, 1, 0]
                                             - 2 * self.prb[1:ci - 1, 1, 0] + self.prb[0:ci - 2, 1, 0]))
        # Corners
        self.pr[0, 0] = self.prt[1, 1, 1]
        self.pr[0, ci - 1] = self.prt[ci - 2, 1, 1]
        self.pr[ri - 1, 0] = self.prb[1, 1, 1]
        self.pr[ri - 1, ci - 1] = self.prb[ci - 2, 1, 1]

        # Store boundary values
        for i in range(0, 2, 1):
            self.prl[0:ri, i, 1] = self.prl[0:ri, i, 0]
            self.prl[0:ri, i, 0] = self.pr[0:ri, i]
            self.prr[0:ri, i, 1] = self.prr[0:ri, i, 0]
            self.prr[0:ri, i, 0] = self.pr[0:ri, ci - 1 - i]
            self.prt[0:ci, i, 1] = self.prt[0:ci, i, 0]
            self.prt[0:ci, i, 0] = self.pr[i, 0:ci]
            self.prb[0:ci, i, 1] = self.prb[0:ci, i, 0]
            self.prb[0:ci, i, 0] = self.pr[ri - 1 - i, 0:ci]

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

    # Webcam image cleanup from grayscale to BW
    blurimg = cv2.medianBlur(img, 21)
    threshimg = cv2.adaptiveThreshold(blurimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    img = threshimg / 256.0

    # Clean up edges of webcam image
    img[0:5, 0:columns] = 1
    img[rows - 5:rows, 0:columns] = 1
    img[0:rows, 0:5] = 1
    img[0:rows, columns - 5:columns] = 1

    # Create rigid material from black portions of image
    imgtemp = img[0:rows - 1, 0:columns]
    idx = imgtemp < 0.5
    fs.mvx[idx] = 1
    imgtemp = img[0:rows, 0:columns - 1]
    idx = imgtemp < 0.5
    fs.mvy[idx] = 1

    # Update image with FDTD solution
    fs.fdtd_update_pr()
    fs.source(tc)
    fs.boundary()
    fs.fdtd_update_v()
    fs.update_domain()
    imgdisp = img + fs.pr

    tc = tc + 1

    # Display image
    cv2.imshow("frame", imgdisp)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
