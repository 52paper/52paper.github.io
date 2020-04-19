After two long discussions, we have got some insights about GAN and reached many agreements. Let's output a tech reports to summarize.

Here is my suggested outline for this report.

1. Brief introduction to GAN, basic idea, formula of GAN.
2. GAN's characteristics, especially comparing with MLE and RL.
3. GAN for Text's main model problems and possible solutions
3.1 gradient vanish problem, how could we solve it
3.2 mode collapse, solutions
3.3 not differentiable for text, three solutions
4. GAN applied in Text, could organized by scenarios, comparing with successfully CV cases.
5. GAN's drawbaks, and why GAN for text is hard
5.1 GAN is hard to tune
5.2 Text is discrete, and hard to be represented in a continous space
5.3 others
6. Future work and in what directions we could improve GAN and use GAN in text


gan这么厉害，我还有的一些疑问：
1. application：看起来很多cv上比较好的应用在text都有体现，虽然效果可能不好，但是有很多都是第一次实现的，比如style transfer, textgan, maskgan，还有没有其他可以挖掘的？
2. 从textgan上看其实gan的可改造性比较强，这种可改造性还是比较适合发paper和有新的application，可以总结一下大家改进和应用的思路？
3. 看起来gan for dialog是一个比mle更适合的解决方案，更合适解决1-n的问题，但是还是很难调的，然后？
