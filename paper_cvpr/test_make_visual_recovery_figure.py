"""CPU-only source-selection and figure regression checks; no generated files."""

import copy
import io
import json
import unittest

import matplotlib.pyplot as plt
import numpy as np

import make_visual_recovery_figure as figure


class VisualRecoveryFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance, cls.control = [json.loads(p.read_bytes()) for p in figure.SOURCES]

    def test_forward_not_heldout(self):
        rows, n, seeds = figure.validate_sources(self.provenance, self.control)
        self.assertEqual((n, seeds), (1000, 3))
        own = rows["own_repair_minus_fault"]
        self.assertEqual(f"{own['delta']:.5f}", "0.06622")
        self.assertNotEqual(own["delta"], self.control["pooled"]["repair"]["heldout"]["delta"])
        self.assertEqual([f"{v:.5f}" for v in rows["donor_repair_minus_fault"]["ci95"]],
                         ["-0.01695", "-0.00959"])
        self.assertEqual(f"{rows['interaction']['delta']:.5f}", "0.07952")

    def test_rejects_inconsistent_sources(self):
        for change in ("ci", "shape", "nonfinite", "interaction", "null"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.provenance)
                own = data["comparisons"]["own_repair_minus_fault"]
                if change == "ci":
                    own["ci95"].reverse()
                elif change == "shape":
                    own["per_image"]["inv"].pop()
                elif change == "nonfinite":
                    own["delta"] = float("nan")
                elif change == "interaction":
                    data["comparisons"]["interaction"]["per_image"]["inv"].reverse()
                else:
                    data["all_mean_null_invariance_passed"] = False
                with self.assertRaises(ValueError):
                    figure.validate_sources(data, self.control)

    def test_layout_and_null_has_no_errorbar(self):
        rows, n, seeds = figure.validate_sources(self.provenance, self.control)
        fig = figure.make_figure(rows)
        try:
            np.testing.assert_array_equal(fig.get_size_inches(), [3.25, 1.8])
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(len(fig.axes[0].containers), 2)
            self.assertEqual(fig.axes[0].lines[-1].get_markerfacecolor(), "none")
            self.assertEqual([text.get_text() for text in fig.axes[0].get_yticklabels()],
                             ["Correct image", "Different image", "No image signal"])
            self.assertEqual(len(fig.texts), 1)
            self.assertEqual([text.get_text() for text in fig.axes[0].texts],
                             ["+0.066", "-0.013", "0"])
            self.assertEqual(figure.STYLE["pdf.fonttype"], 42)
            with plt.rc_context(figure.STYLE):
                pdf = io.BytesIO()
                fig.savefig(pdf, format="pdf", transparent=True)
            self.assertIn(b"/FontFile2", pdf.getvalue())
            self.assertNotIn(b"/Subtype /Type3", pdf.getvalue())
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in [*fig.texts, *fig.axes[0].texts,
                         *fig.axes[0].get_xticklabels(), *fig.axes[0].get_yticklabels()]:
                box = text.get_window_extent(renderer)
                self.assertTrue(fig.bbox.contains(box.x0, box.y0), text.get_text())
                self.assertTrue(fig.bbox.contains(box.x1, box.y1), text.get_text())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
