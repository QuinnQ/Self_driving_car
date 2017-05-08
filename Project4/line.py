import numpy as np
img_size = (1280, 720)

# meters per pixel in y and x dimension
ym_per_pix = 30/720
xm_per_pix = 3.7/700


class Line:
    def __init__(self, iters=5):
        self.niters = iters
        # Was the line detected in last iteration
        self.detected = False

        # x values
        self.currentx_fit = None
        self.recentx_fit = []
        self.avgx_fit = None
        self.yfit = np.linspace(0, img_size[1]-1, img_size[1])

        # polynomial coefficients
        self.current_fit = None
        self.recent_fits = []
        self.avg_fit = None

        self.radius = None
        self.base_pos = None
        # x and y values for detected line pixels
        self.allx = None
        self.ally = None

    def add_line(self, x_pixels, y_pixels):
        self.detected = True
        self.allx = x_pixels
        self.ally = y_pixels
        self.cal_fit()  # current_fit, currentx_fit, (if avgx_fit is None: recent_fits, avg_fit, recentx_fit, avgx_fit)
        self.cal_radius(self.currentx_fit)
        self.cal_base_pos(self.current_fit)

    def update(self):
        if len(self.recent_fits) >= self.niters:
            self.recent_fits.pop(0)
            self.recentx_fit.pop(0)

        self.recent_fits.append(self.current_fit)
        self.avg_fit = np.average(np.array(self.recent_fits), axis=0)

        self.recentx_fit.append(self.currentx_fit)
        self.avgx_fit = np.average(np.array(self.recentx_fit), axis=0)
        self.cal_radius(self.avgx_fit)
        self.detected = False

    def cal_radius(self, x_fit):
        y_eval = np.max(self.yfit)
        fit_cr = np.polyfit(self.yfit*ym_per_pix, x_fit*xm_per_pix, 2)
        self.radius = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5)/np.absolute(2*fit_cr[0])

    def cal_fit(self):
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.currentx_fit = self.current_fit[0] * (self.yfit ** 2) + self.current_fit[1] * self.yfit + self.current_fit[2]
        if self.avgx_fit is None:
            self.update()

    def cal_base_pos(self, fit):
        y = img_size[1]
        x = fit[0]*y**2 + fit[1]*y + fit[2]
        self.base_pos = x*xm_per_pix
        return x*xm_per_pix

    def cal_distance(self, other):
        return abs(self.base_pos-other.base_pos)

    def check_xfitted_diff(self):
        return np.linalg.norm(self.currentx_fit - self.avgx_fit)

    def check_curves_diff(self, other):
        r1 = self.radius
        r2 = other.radius
        if max(abs(r1), abs(r2)) > 2000:
            return False
        else:
            return (r1 * r2 < 0)

